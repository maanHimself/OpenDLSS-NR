#include "nr_model.h"

#include <cassert>
#include <fstream>
#include <sstream>

#include "json.h"
#include "numeric.h"
#include "sha256.h"

namespace nr {

uint32_t packedInputIndex(uint32_t k) {
  uint32_t base = k & ~31u;
  uint32_t within = k & 31u;
  uint32_t half = within & 16u;
  uint32_t quarter = within & 15u;
  return base + half + (quarter >> 2) * 2 + (quarter & 1) + (((quarter & 2) != 0) ? 8 : 0);
}

uint32_t inversePackedInputIndex(uint32_t k) {
  uint32_t base = k & ~31u;
  uint32_t within = k & 31u;
  return base + (within & 17u) + ((within & 2u) << 1) + ((within & 4u) << 1) + ((within & 8u) >> 2);
}

uint32_t packedWeightIndex(uint32_t k, uint32_t n, uint32_t outputChannels) {
  uint32_t kTile = k >> 5, kIn = k & 31;
  uint32_t nTile = n >> 7, nIn = n & 127;
  uint32_t nHalf = nIn >> 6, nGroup = (nIn & 63) >> 4, nInGroup = nIn & 15;
  uint32_t lane = ((nInGroup & 7) << 2) | ((kIn & 15) >> 2);
  uint32_t byteInLane = ((nInGroup >> 3) << 3) | ((kIn >> 4) << 2) | (kIn & 3);
  return kTile * outputChannels * 32 + nTile * 4096 + nHalf * 2048 + nGroup * 512 + lane * 16 + byteInLane;
}

namespace {
std::string readText(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read " + path);
  std::stringstream stream;
  stream << file.rdbuf();
  return stream.str();
}

std::vector<uint8_t> readBinary(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("cannot read " + path);
  std::streamsize size = file.tellg();
  file.seekg(0);
  std::vector<uint8_t> bytes((size_t)size);
  file.read(reinterpret_cast<char*>(bytes.data()), size);
  return bytes;
}

// Half index of an f16 weight inside a packed matrix: 16x16 tiles in (k, n) row-major order, each tile one
// m16n8k16 B fragment pair - lane (n % 8) * 4 + (k % 8) / 2, four halves per lane per 8-column half.
// The two f16 matrices of the network (the 16 -> 32 input adapter and the 32 -> 4 head) are the two ways this
// tiling degenerates: one k tile by two n tiles, and two k tiles by one n tile.
uint32_t packedF16WeightIndex(uint32_t inputChannel, uint32_t outputChannel, uint32_t outputChannels) {
  const uint32_t nTiles = (outputChannels + 15) / 16;
  const uint32_t tile = (inputChannel >> 4) * nTiles + (outputChannel >> 4);
  const uint32_t k = inputChannel & 15, n = outputChannel & 15;
  const uint32_t lane = ((n & 7) << 2) | ((k & 7) >> 1);
  const uint32_t fragment = (k >= 8 ? 2 : 0) + (k & 1);
  return tile * 256 + lane * 8 + ((n >> 3) & 1) * 4 + fragment;
}

// Natural window token (row-major in the 8x8 window) -> physical token (4x4 tiles of 16).
uint32_t tiledToken(uint32_t token) {
  uint32_t x = token & 7, y = token >> 3;
  return (y >> 2) * 32 + (x >> 2) * 16 + (y & 3) * 4 + (x & 3);
}
}  // namespace

Model::Model(vk::Context& context, const std::string& directory, bool verifyHashes) : context_(context) {
  // A throw destroys what was made so far (the destructor does not run for a constructor that throws).
  try {
    json::Value manifest = json::parse(readText(directory + "/manifest.json"));
    blockCount_ = (uint32_t)manifest["totals"]["blockCount"].integer();
    for (const json::Value& stage : manifest["stages"].array) {
      Stage loaded;
      loaded.id = stage["id"].str();
      loaded.bytes = readBinary(directory + "/model/" + stage["file"].str());
      if (loaded.bytes.size() != (size_t)stage["packedByteLength"].integer())
        throw std::runtime_error("stage size mismatch: " + loaded.id);
      if (verifyHashes) {
        std::string digest = sha256Hex(loaded.bytes.data(), loaded.bytes.size());
        if (digest != stage["sha256"].str()) throw std::runtime_error("stage SHA-256 mismatch: " + loaded.id);
      }
      stages_.push_back(std::move(loaded));
    }
    for (const json::Value& entry : manifest["tensors"].array) {
      Tensor tensor;
      tensor.name = entry["name"].str();
      tensor.block = (int)entry["block"].integer();
      tensor.layer = (int)entry["layer"].integer();
      tensor.parameter = entry["parameter"].str();
      tensor.stage = entry["stage"].str();
      tensor.stageOffset = (uint32_t)entry["stageOffset"].integer();
      tensor.byteLength = (uint32_t)entry["byteLength"].integer();
      const Stage* stage = nullptr;
      for (const Stage& candidate : stages_) if (candidate.id == tensor.stage) stage = &candidate;
      if (!stage) throw std::runtime_error("tensor references unknown stage " + tensor.stage);
      if ((size_t)tensor.stageOffset + tensor.byteLength > stage->bytes.size())
        throw std::runtime_error("tensor exceeds stage " + tensor.name);
      tensor.bytes = stage->bytes.data() + tensor.stageOffset;
      // Raw bytes on the GPU for per-column aux vectors (padded to 4 bytes).
      size_t padded = (tensor.byteLength + 3) & ~3u;
      std::vector<uint8_t> paddedBytes(padded, 0);
      memcpy(paddedBytes.data(), tensor.bytes, tensor.byteLength);
      // Owned by tensors_ before anything that can throw, so a failure releases it.
      const std::string name = tensor.name;
      Tensor& stored = tensors_[name] = std::move(tensor);
      stored.raw = context_.createBuffer(padded, false, "tensor raw");
      context_.upload(stored.raw, paddedBytes.data(), padded);
    }
    for (uint32_t k = 0; k < 64; ++k) {
      if (inversePackedInputIndex(packedInputIndex(k)) != k)
        throw std::runtime_error("chained index permutation is not invertible");
    }
  } catch (...) {
    release();
    throw;
  }
}

Model::~Model() { release(); }

void Model::release() {
  for (auto& [name, buffer] : matrices_) context_.destroyBuffer(buffer);
  for (auto& [name, tensor] : tensors_) context_.destroyBuffer(tensor.raw);
  matrices_.clear();
  tensors_.clear();
}

const Tensor& Model::tensor(int block, int layer, const std::string& parameter) const {
  std::string name = "block" + std::to_string(block) + ".layer" + std::to_string(layer) + "." + parameter;
  auto it = tensors_.find(name);
  if (it == tensors_.end()) throw std::runtime_error("missing tensor " + name);
  return it->second;
}

std::vector<uint8_t> Model::fp8MatrixBytes(const Tensor& tensor, uint32_t byteOffset, uint32_t K, uint32_t Nmatrix,
                                           bool swizzleK, uint32_t batchK, bool tileMajor,
                                           const std::vector<uint32_t>* columnSource) {
  if (batchK == 0) batchK = K;
  std::string key = tensor.name + "/" + std::to_string(byteOffset) + "/" + std::to_string(K) + "x" + std::to_string(Nmatrix);
  if (K % 32 || Nmatrix % 16 || batchK % 32 || K % batchK)
    throw std::runtime_error("FP8 matrix shape must be K%32==0, N%16==0, batchK | K: " + key);
  size_t bytes = (size_t)K * Nmatrix;
  if ((size_t)byteOffset + bytes > tensor.byteLength)
    throw std::runtime_error("FP8 matrix exceeds tensor " + key);
  std::vector<uint8_t> plain(bytes);
  for (uint32_t j = 0; j < K; ++j) {
    uint32_t k = swizzleK ? inversePackedInputIndex(j) : j;
    for (uint32_t n = 0; n < Nmatrix; ++n) {
      const uint32_t sourceColumn = columnSource ? (*columnSource)[n] : n;
      uint8_t code = tensor.bytes[byteOffset + packedWeightIndex(k, sourceColumn, Nmatrix)];
      if ((code & 0x7f) == 0x7f) { code = 0; ++nanWeights_; }  // E4M3FN NaN decodes to 0 in the reference
      // The chained permutation stays within 32-groups, so it commutes with the batch split.
      uint32_t kk = j % batchK;
      if (tileMajor) plain[(((size_t)(j / batchK) * (batchK / 32) + kk / 32) * Nmatrix + n) * 32 + (kk % 32)] = code;
      else plain[((size_t)(j / batchK) * Nmatrix + n) * batchK + kk] = code;
    }
  }
  return plain;
}

const vk::Buffer& Model::fp8Matrix(const Tensor& tensor, uint32_t byteOffset, uint32_t K, uint32_t Nmatrix,
                                   bool swizzleK, uint32_t batchK, bool tileMajor) {
  if (batchK == 0) batchK = K;
  std::string key = tensor.name + (tileMajor ? "/fp8K32/" : "/fp8T/") + std::to_string(byteOffset) + "/" + std::to_string(K) +
                    "x" + std::to_string(Nmatrix) + "/" + std::to_string(batchK) + (swizzleK ? "/s" : "/n");
  auto it = matrices_.find(key);
  if (it != matrices_.end()) return it->second;
  std::vector<uint8_t> plain = fp8MatrixBytes(tensor, byteOffset, K, Nmatrix, swizzleK, batchK, tileMajor);
  vk::Buffer buffer = context_.createBuffer(plain.size(), false, "fp8 weights");
  context_.upload(buffer, plain.data(), plain.size());
  return matrices_[key] = buffer;
}

const vk::Buffer& Model::fp8MatrixPermuted(const Tensor& tensor, uint32_t byteOffset, uint32_t K, uint32_t Nmatrix,
                                           uint32_t batchK, const std::vector<uint32_t>& columnSource, const std::string& tag) {
  std::string key = tensor.name + "/fp8perm/" + tag + "/" + std::to_string(byteOffset) + "/" + std::to_string(K) + "x" +
                    std::to_string(Nmatrix) + "/" + std::to_string(batchK);
  auto it = matrices_.find(key);
  if (it != matrices_.end()) return it->second;
  if (columnSource.size() != Nmatrix) throw std::runtime_error("column permutation size mismatch: " + key);
  std::vector<uint8_t> plain = fp8MatrixBytes(tensor, byteOffset, K, Nmatrix, true, batchK, true, &columnSource);
  vk::Buffer buffer = context_.createBuffer(plain.size(), false, "fp8 weights (permuted)");
  context_.upload(buffer, plain.data(), plain.size());
  return matrices_[key] = buffer;
}

const vk::Buffer& Model::f16Matrix(const Tensor& tensor, uint32_t byteOffset, uint32_t K, uint32_t N,
                                   uint32_t& paddedN) {
  paddedN = (N + 15) & ~15u;
  std::string key = tensor.name + "/f16/" + std::to_string(byteOffset) + "/" + std::to_string(K) + "x" +
                    std::to_string(N);
  auto it = matrices_.find(key);
  if (it != matrices_.end()) return it->second;
  if (K % 16) throw std::runtime_error("f16 matrix K must be a multiple of 16");
  std::vector<uint16_t> plain((size_t)K * paddedN, 0);
  for (uint32_t k = 0; k < K; ++k) {
    for (uint32_t n = 0; n < N; ++n) {
      uint32_t halfIndex = (byteOffset >> 1) + packedF16WeightIndex(k, n, N);
      const uint8_t* p = tensor.bytes + halfIndex * 2;
      if (halfIndex * 2 + 1 >= tensor.byteLength) throw std::runtime_error("f16 matrix exceeds tensor " + key);
      plain[(size_t)k * paddedN + n] = (uint16_t)(p[0] | (p[1] << 8));
    }
  }
  vk::Buffer buffer = context_.createBuffer(plain.size() * 2, false, "f16 weights");
  context_.upload(buffer, plain.data(), plain.size() * 2);
  return matrices_[key] = buffer;
}

const vk::Buffer& Model::relativeBias(const Tensor& tensor, uint32_t relativeByteOffset, uint32_t heads) {
  std::string key = tensor.name + "/prior/" + std::to_string(relativeByteOffset) + "/" + std::to_string(heads);
  auto it = matrices_.find(key);
  if (it != matrices_.end()) return it->second;
  std::vector<uint16_t> prior((size_t)heads * 64 * 64);
  // Native stores the bias with both tokens in physical order; the kernels want the query in natural order (the
  // A rows of S = Q K^T) and the key in physical order (the B columns, which is how K / V are staged).
  for (uint32_t head = 0; head < heads; ++head) {
    for (uint32_t query = 0; query < 64; ++query) {
      for (uint32_t physical = 0; physical < 64; ++physical) {
        uint32_t q = tiledToken(query), k = physical;
        uint32_t m = q & 15, n = k & 15;
        uint32_t lane = ((m & 7) << 2) | ((n & 7) >> 1);
        uint32_t fragment = (m >= 8 ? 2 : 0) + (n & 1);
        uint32_t tileOffset = (q >> 4) * 1024 + (k >> 4) * 256;
        uint32_t laneOffset = lane * 8 + (n >> 3) * 4;
        uint32_t halfIndex = tileOffset + laneOffset + fragment;
        uint32_t byteIndex = relativeByteOffset + head * 8192 + halfIndex * 2;
        if (byteIndex + 1 >= tensor.byteLength) throw std::runtime_error("relative bias exceeds tensor " + key);
        prior[((size_t)head * 64 + query) * 64 + physical] =
            (uint16_t)(tensor.bytes[byteIndex] | (tensor.bytes[byteIndex + 1] << 8));
      }
    }
  }
  vk::Buffer buffer = context_.createBuffer(prior.size() * 2, false, "attention prior");
  context_.upload(buffer, prior.data(), prior.size() * 2);
  return matrices_[key] = buffer;
}

}  // namespace nr

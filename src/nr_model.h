// Model loading: manifest, packed E4M3 stage files and the host-side re-layout of the MMA-fragment-ordered
// weights into plain matrices that cooperative-matrix loads consume directly.
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vk_context.h"

namespace nr {

struct Tensor {
  std::string name;
  int block = 0;
  int layer = 0;
  std::string parameter;
  std::string stage;
  uint32_t stageOffset = 0;
  uint32_t byteLength = 0;
  const uint8_t* bytes = nullptr;  // into the owning stage
  vk::Buffer raw;                  // raw packed bytes on the GPU (aux vectors, scales)
};

// Native within-32 chained activation index used by every FP8 GEMM A operand.
uint32_t packedInputIndex(uint32_t k);
uint32_t inversePackedInputIndex(uint32_t k);
// Byte index of weight (k, n) inside a packed FP8 matrix of N columns, as the model file stores it.
uint32_t packedWeightIndex(uint32_t k, uint32_t n, uint32_t outputChannels);

class Model {
 public:
  Model(vk::Context& context, const std::string& directory, bool verifyHashes = true);
  ~Model();

  const Tensor& tensor(int block, int layer = 0, const std::string& parameter = "layer") const;

  // Plain [K][Nmatrix] E4M3 matrix for coopmat B loads. When swizzleK is set the
  // K rows are permuted by the inverse chained index so that A operands load in
  // natural channel order (the permutation stays inside each 16-product group).
  // Plain E4M3 weights, k32-tile-major for the GEMM: [K / batchK][batchK / 32][Nmatrix][32].
  // batchK = 0 means a single batch (the whole K).
  // tileMajor = false: [K / batchK][Nmatrix][batchK] (N-major, K contiguous) for single-multiply B operands.
  const vk::Buffer& fp8Matrix(const Tensor& tensor, uint32_t byteOffset, uint32_t K, uint32_t Nmatrix,
                              bool swizzleK = true, uint32_t batchK = 0, bool tileMajor = true);
  // The same bytes without a buffer (for packed multi-matrix buffers).
  std::vector<uint8_t> fp8MatrixBytes(const Tensor& tensor, uint32_t byteOffset, uint32_t K, uint32_t Nmatrix,
                                      bool swizzleK = true, uint32_t batchK = 0, bool tileMajor = true,
                                      const std::vector<uint32_t>* columnSource = nullptr);
  // fp8Matrix with the output columns permuted: plain column c holds tensor column columnSource[c].
  const vk::Buffer& fp8MatrixPermuted(const Tensor& tensor, uint32_t byteOffset, uint32_t K, uint32_t Nmatrix,
                                      uint32_t batchK, const std::vector<uint32_t>& columnSource, const std::string& tag);
  // Plain [K][Npadded] f16 matrix (the pre adapter and the post head).
  const vk::Buffer& f16Matrix(const Tensor& tensor, uint32_t byteOffset, uint32_t K, uint32_t N, uint32_t& paddedN);
  // Learned 64x64 attention prior per head as f16 [heads][64 query][64 physical key].
  const vk::Buffer& relativeBias(const Tensor& tensor, uint32_t relativeByteOffset, uint32_t heads);

  size_t nanWeightsReplaced() const { return nanWeights_; }
  uint32_t blockCount() const { return blockCount_; }

 private:
  void release();   // the destructor's cleanup (also a constructor's that throws)
  struct Stage {
    std::string id;
    std::vector<uint8_t> bytes;
  };
  vk::Context& context_;
  std::vector<Stage> stages_;
  std::map<std::string, Tensor> tensors_;
  std::map<std::string, vk::Buffer> matrices_;
  size_t nanWeights_ = 0;
  uint32_t blockCount_ = 0;
};

// Exact float16 load of the aux (per-column) vectors stored inside a tensor.
inline uint16_t auxHalf(const Tensor& tensor, uint32_t byteOffset, uint32_t column) {
  const uint8_t* p = tensor.bytes + byteOffset + column * 2;
  return (uint16_t)(p[0] | (p[1] << 8));
}
inline float auxF32(const Tensor& tensor, uint32_t byteOffset) {
  float value;
  memcpy(&value, tensor.bytes + byteOffset, 4);
  return value;
}

}  // namespace nr

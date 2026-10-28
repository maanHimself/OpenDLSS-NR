#include "nr_graph.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace nr {

namespace {
uint32_t alignUp(uint32_t value, uint32_t alignment) { return (value + alignment - 1) / alignment * alignment; }
uint32_t standardHidden(uint32_t channels) {
  if (channels == 32 || channels == 64 || channels == 128 || channels == 256) return 128;
  throw std::runtime_error("no fused layout for " + std::to_string(channels) + " channels");
}
// The four window views of the shifted-window attention, as origin offsets (-shiftX, -shiftY). Each resolution
// level cycles through them, one step per block at that level (Graph::takeWindowPhase).
void windowPhase(uint32_t index, uint32_t& shiftX, uint32_t& shiftY) {
  static const uint32_t phases[4][2] = {{0, 0}, {4, 4}, {4, 0}, {0, 4}};
  shiftX = phases[index & 3][0];
  shiftY = phases[index & 3][1];
}
}  // namespace

// Every level halves its input and rounds the result up to 4; the decoder doubles the chain back up. A
// dimension therefore needs enough headroom for all of those halvings to be exact, which is what the padded
// ("full") field provides: it is aligned to two to the power of the number of size reductions the graph makes
// on that axis. Six of them are the halvings themselves; level 0 adds a seventh when it is not a whole number
// of 8-pixel windows, because the decoder's level-0 upsample produces whole windows and is then cropped back.
static uint32_t fieldAlignment(uint32_t valid) {
  uint32_t reductions = 0, size = valid;
  for (int level = 0; level < 6; ++level) {
    const uint32_t half = alignUp((size + 1) / 2, 4);
    if (half < size) ++reductions;
    if (level == 0 && half % 8 != 0) ++reductions;
    size = half;
  }
  return 1u << reductions;
}

Geometry Geometry::fromValid(uint32_t validWidth, uint32_t validHeight) {
  Geometry g;
  g.validWidth = validWidth;
  g.validHeight = validHeight;
  const uint32_t alignWidth = fieldAlignment(validWidth), alignHeight = fieldAlignment(validHeight);
  g.fullWidth = std::max(320u, alignUp(validWidth, alignWidth));
  g.fullHeight = std::max(320u, alignUp(validHeight, alignHeight));
  // One more alignment step on the width when both axes are a multiple of four alignments. It is native's
  // rule and there is no stated reason for it; it has to be reproduced because the field size decides the
  // window grid and therefore the result inside the valid rectangle as well.
  if (g.fullWidth % (4 * alignWidth) == 0 && g.fullHeight % (4 * alignHeight) == 0) g.fullWidth += alignWidth;
  uint32_t width = g.fullWidth, height = g.fullHeight;
  for (int level = 0; level < 6; ++level) {
    width = alignUp((width + 1) / 2, 4);
    height = alignUp((height + 1) / 2, 4);
    g.levels[level] = {width, height};
  }
  // The extra reduction above exists so that level 0 is a whole number of 8-pixel windows and the decoder's
  // level-0 upsample can write it directly. The 320 floor can break that for a very small axis (widths 1-16 and
  // 25-32), where the network runs its level-0 decoder stage on alignUp(level0, 8) and crops in the last block;
  // this port does not implement that crop, so it refuses the size rather than differing silently.
  if (g.levels[0].width % 8 || g.levels[0].height % 8)
    throw std::runtime_error("unsupported size " + std::to_string(validWidth) + "x" + std::to_string(validHeight) +
                             ": level 0 (" + std::to_string(g.levels[0].width) + "x" + std::to_string(g.levels[0].height) +
                             ") is not a whole number of 8-pixel windows; use at least 33 pixels on each axis");
  return g;
}

FusedLayout fusedLayout(uint32_t channels, uint32_t base) {
  FusedLayout l;
  l.hidden = standardHidden(channels);
  l.heads = channels / 32;
  l.expand = base;
  l.expertFfn = channels >= 64;
  l.expertCount = l.expertFfn ? channels / 32 : 0;
  uint32_t expandBytes = l.expertFfn ? l.expertCount * channels * 128 : channels * l.hidden;
  l.contractWeights = base + expandBytes;
  uint32_t ffnWeightBytes = l.expertFfn ? expandBytes + l.expertCount * 128 * 32 + l.expertCount * 32 * channels
                                        : expandBytes + l.hidden * channels;
  l.ffnCosSkip = base + ffnWeightBytes + 16;
  l.qkv = l.ffnCosSkip + channels * 2 + 16;
  l.relative = l.qkv + channels * channels * 3;
  l.scale = l.relative + l.heads * 8192;
  l.projection = l.scale + alignUp(l.heads * 4, 16);
  l.attnCosSkip = l.projection + channels * channels;
  l.endWithoutPadding = l.attnCosSkip + channels * 2;
  return l;
}

FusedLayout preFusedLayout() {
  FusedLayout l;
  l.hidden = 128; l.heads = 1;
  l.expand = 0; l.contractWeights = 4096; l.inputAdapter = 8208; l.ffnCosSkip = 9232; l.qkv = 9312;
  l.relative = 12384; l.scale = 20576; l.projection = 20592; l.attnCosSkip = 21616; l.endWithoutPadding = 21680;
  return l;
}

FusedLayout upsampleFusedLayout(uint32_t inputChannels, uint32_t channels) {
  if (inputChannels != channels * 2) throw std::runtime_error("upsample layout expects 2x input channels");
  FusedLayout l;
  l.hidden = standardHidden(channels);
  l.heads = channels / 32;
  uint32_t narrowPadding = channels == 32 ? 16 : 0;
  l.expand = 0;
  l.expertFfn = channels >= 64;
  l.expertCount = l.expertFfn ? channels / 32 : 0;
  uint32_t expandBytes = l.expertFfn ? l.expertCount * channels * 128 : channels * l.hidden;
  l.contractWeights = expandBytes;
  uint32_t ffnWeightBytes = l.expertFfn ? expandBytes + l.expertCount * 128 * 32 + l.expertCount * 32 * channels
                                        : expandBytes + l.hidden * channels;
  l.upsampleWeight = ffnWeightBytes;
  l.ffnCosSkip = l.upsampleWeight + inputChannels * channels + narrowPadding;
  l.transitionScale = l.ffnCosSkip + channels * 2 + narrowPadding;
  l.qkv = l.transitionScale + channels * 2;
  l.relative = l.qkv + channels * channels * 3;
  l.scale = l.relative + l.heads * 8192;
  l.projection = l.scale + alignUp(l.heads * 4, 16);
  l.attnCosSkip = l.projection + channels * channels;
  l.endWithoutPadding = l.attnCosSkip + channels * 2;
  return l;
}

FusedLayout postFusedLayout() {
  FusedLayout l;
  l.hidden = 128; l.heads = 1;
  l.expand = 0; l.contractWeights = 4096; l.ffnCosSkip = 8208; l.inputScale = 8272; l.adapterScale = 8336;
  l.qkv = 8400; l.relative = 11472; l.scale = 19664; l.projection = 19680; l.attnCosSkip = 20704;
  l.postWeights = 20784; l.endWithoutPadding = 21808;
  return l;
}

Graph::Routes Graph::routesFromEnvironment() {
  Routes r;
  r.fusePre = !getenv("DLSS5VK_NO_FUSE_PRE");
  r.fusePool = !getenv("DLSS5VK_NO_FUSE_POOL");
  r.fuseUpres = !getenv("DLSS5VK_NO_FUSE_UPRES");
  r.fusePost = !getenv("DLSS5VK_NO_FUSE_POST");
  r.chainMask = getenv("DLSS5VK_CHAIN_MASK") ? atoi(getenv("DLSS5VK_CHAIN_MASK")) : 3;   // bit 4 measured neutral
  r.deferMax = getenv("DLSS5VK_DEFER_MAX") ? (uint32_t)atoi(getenv("DLSS5VK_DEFER_MAX")) : 128u;
  r.vitChain = getenv("DLSS5VK_VIT_CHAIN") ? atoi(getenv("DLSS5VK_VIT_CHAIN")) != 0 : true;
  return r;
}

Graph::Graph(vk::Context& context, Model& model, Kernels& kernels, const Geometry& geometry, Options options)
    : context_(context), model_(model), kernels_(kernels), geometry_(geometry), options_(options),
      routes_(routesFromEnvironment()) {
  // Chaining links consecutive PTX launches through device counters and drops the barrier between them, so every
  // launch in the chain must take its PTX route. Any switch that sends one kernel family (or one fused block)
  // back to GLSL takes the whole graph back to barriers.
  routes_.chain = Kernels::chainEnabled() && options_.fusedBlocks && !options_.captureIntermediates &&
                  Kernels::ptxGemmEnabled() && Kernels::ptxFfnEnabled() && Kernels::ptxQkvEnabled() &&
                  Kernels::ptxBlock32Enabled() && routes_.fusePre && routes_.fusePool && routes_.fuseUpres &&
                  routes_.fusePost;
  // The ViT launches are their own chain (a single completion counter per GEMM); they need only the ViT routes.
  routes_.vitChain = routes_.vitChain && Kernels::chainEnabled() && options_.fusedBlocks &&
                     !options_.captureIntermediates && Kernels::ptxGemmVEnabled();
  // A chain's counters are indexed by pixel-row band or window row of the tallest stage, which is the field
  // itself (blocks 0 and 70). Past the region's capacity the indices would run into the next region.
  const uint32_t tallestRow = (geometry_.fullHeight + 4 + 7) / 8;   // + the largest window shift
  if (tallestRow > Kernels::kSyncCountersPerRegion) {
    fprintf(stderr, "warning: field height %u needs %u chaining counters per block (limit %u); running with barriers\n",
            geometry_.fullHeight, tallestRow, Kernels::kSyncCountersPerRegion);
    routes_.chain = routes_.vitChain = false;
  }
  if (model_.blockCount() != kBlockCount)
    throw std::runtime_error("the model has " + std::to_string(model_.blockCount()) + " blocks; this graph is the " +
                             std::to_string(kBlockCount) + "-block network");
}

Graph::~Graph() {
  for (auto& activation : activations_) context_.destroyBuffer(activation->buffer);
  context_.freeMemory(activationBlock_);   // nothing is bound to it now
}

// The lifetimes are only as complete as the reports: every use the recording makes of a buffer, transfers (the
// captures' copies, fills) included, must reach the use tracker (vk::Context::cmdCopyBuffer, cmdFillZero,
// allocateSet, deviceAddress), or shareActivations takes that buffer for unused or for dead too early.
void Graph::trackLifetimes(bool on) {
  if (!on) {
    context_.setUseTracker(nullptr);
    return;
  }
  lifetimes_.clear();
  context_.setUseTracker([this](VkBuffer buffer, uint32_t epoch) {
    auto [it, inserted] = lifetimes_.try_emplace(buffer, epoch, epoch);
    if (!inserted) {
      it->second.first = std::min(it->second.first, epoch);
      it->second.second = std::max(it->second.second, epoch);
    }
  });
}

void Graph::rebind(Activation& activation, vk::Buffer buffer) {
  context_.destroyBuffer(activation.buffer);
  activation.buffer = buffer;
}

std::pair<VkDeviceSize, VkDeviceSize> Graph::shareActivations(std::initializer_list<const Activation*> keep) {
  if (activationBlock_) throw std::runtime_error("shareActivations: already shared");
  if (lifetimes_.empty()) throw std::runtime_error("shareActivations: no recording was tracked (trackLifetimes)");
  struct Placement {
    Activation* activation;
    uint32_t first, last;
    VkDeviceSize offset, bytes;
  };
  std::vector<Placement> mine;
  VkDeviceSize before = 0, block = 0, alignment = 256;
  uint32_t types = ~0u;
  for (auto& owned : activations_) {
    Activation* a = owned.get();
    if (a == head_ || std::find(keep.begin(), keep.end(), a) != keep.end()) continue;
    // The captures are read after the run, like the head. Their copies' sources are placed like any activation,
    // alive until the copy reads them (a use the tracked recording reported).
    if (std::any_of(boundaries_.begin(), boundaries_.end(), [a](const auto& c) { return c.second == a; })) continue;
    before += a->buffer.size;
    auto it = lifetimes_.find(a->buffer.buffer);
    if (it == lifetimes_.end()) {
      rebind(*a, context_.createBuffer(16, false, a->label.c_str()));
      continue;
    }
    const VkMemoryRequirements r = context_.bufferRequirements(a->buffer.size);
    alignment = std::max(alignment, r.alignment);
    types &= r.memoryTypeBits;
    mine.push_back({a, it->second.first, it->second.second, 0, r.size});
  }
  lifetimes_.clear();
  std::stable_sort(mine.begin(), mine.end(), [](const Placement& a, const Placement& b) { return a.bytes > b.bytes; });
  std::vector<Placement> placed;
  for (Placement p : mine) {
    // The lowest offset clear of every placed activation alive in an overlapping interval.
    std::vector<std::pair<VkDeviceSize, VkDeviceSize>> taken;
    for (const Placement& q : placed)
      if (q.first <= p.last && p.first <= q.last) taken.emplace_back(q.offset, q.offset + q.bytes);
    std::sort(taken.begin(), taken.end());
    VkDeviceSize offset = 0;
    for (const auto& [start, end] : taken) {
      if (offset + p.bytes <= start) break;
      offset = std::max(offset, (end + alignment - 1) / alignment * alignment);
    }
    p.offset = offset;
    block = std::max(block, offset + p.bytes);
    placed.push_back(p);
  }
  if (placed.empty()) return {before, 0};
  activationBlock_ = context_.allocateDeviceMemory(block, types);
  for (const Placement& p : placed)
    rebind(*p.activation, context_.createBufferAt(activationBlock_, p.offset, p.bytes, p.activation->label.c_str()));
  vk::Buffer whole = context_.createBufferAt(activationBlock_, 0, block, "activation block");
  context_.fillZero(whole);
  context_.destroyBuffer(whole);
  return {before, block};
}

Activation* Graph::allocate(const std::string& label, uint32_t rows, uint32_t channels, Format format) {
  // Re-recording the graph reuses the buffers allocated by the first pass.
  std::string key = label + "/" + std::to_string(rows) + "x" + std::to_string(channels) + "/" + std::to_string((int)format);
  auto existing = allocationsByKey_.find(key);
  if (existing != allocationsByKey_.end()) {
    if (!usedThisRecord_.insert(key).second) throw std::runtime_error("duplicate activation label " + label);
    return existing->second;
  }
  if (!usedThisRecord_.insert(key).second) throw std::runtime_error("duplicate activation label " + label);
  auto activation = std::make_unique<Activation>();
  activation->format = format;
  activation->rows = rows;
  activation->channels = channels;
  activation->allocRows = alignRows(rows);
  activation->label = label;
  VkDeviceSize bytes = (VkDeviceSize)activation->allocRows * channels * formatBytes(format);
  activation->buffer = context_.createBuffer(bytes, false, activation->label.c_str());
  context_.fillZero(activation->buffer);
  activations_.push_back(std::move(activation));
  allocationsByKey_[key] = activations_.back().get();
  return activations_.back().get();
}

const std::vector<std::string>& Graph::referenceBoundaryNames() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> list;
    const int transitionsAfter[] = {0, 4, 8, 14, 22};
    for (int block = 0; block <= 69; ++block) {
      list.push_back("block-" + std::to_string(block));
      if (std::find(std::begin(transitionsAfter), std::end(transitionsAfter), block) != std::end(transitionsAfter))
        list.push_back("transition-" + std::to_string(block) + "-" + std::to_string(block + 1));
    }
    return list;
  }();
  return names;
}

void Graph::capture(VkCommandBuffer commands, const std::string& name, const Activation& source) {
  if (!options_.captureBoundaries) return;
  Activation* copy = allocate("boundary " + name, source.rows, source.channels, source.format);
  VkBufferCopy region{0, 0, source.validBytes()};
  context_.transferBarrier(commands);
  // Through the context, so the lifetimes see the copy: the source is still read here, after its last kernel, and
  // the capture is written here.
  context_.cmdCopyBuffer(commands, source.buffer, copy->buffer, region);
  context_.transferBarrier(commands);
  boundaries_[name] = copy;
}

Graph::Temporaries Graph::createTemporaries(const std::string& label, uint32_t rows, uint32_t channels) {
  Temporaries t;
  FusedLayout layout = fusedLayout(channels);
  uint32_t hidden = layout.expertFfn ? layout.expertCount * 128 : layout.hidden;
  const bool fused = options_.fusedBlocks && !options_.captureIntermediates;
  if (fused && channels == 32) return t;   // the fused 32-channel block keeps everything on chip
  if (!fused) t.ffn = allocate(label + " FFN", rows, hidden, Format::E4);
  if (layout.expertFfn) t.ffnNarrow = allocate(label + " FFN narrow", rows, channels, Format::E4);
  if (!fused) t.ffnResidual = allocate(label + " FFN residual", rows, channels, Format::F16);
  t.ffnQuantized = allocate(label + " FFN quantized", rows, channels, Format::E4);
  // c256 (one workgroup per SM) is faster with the separate projection GEMM, hence the default width limit.
  if (fused && layout.expertFfn && Kernels::ptxFfnEnabled() && channels <= routes_.deferMax)
    t.ffnQuantized2 = allocate(label + " FFN quantized B", rows, channels, Format::E4);
  if (!fused) t.qkv = allocate(label + " QKV", rows, channels * 3, Format::F16);
  if (!fused) t.normalized = allocate(label + " normalized QKV", rows, channels * 3, Format::E4);
  t.attended = allocate(label + " attended", rows, channels, Format::E4);
  return t;
}

Graph::SplitTemporaries Graph::createSplitTemporaries(const std::string& label, uint32_t rows) {
  SplitTemporaries t;
  t.branch = allocate(label + " split branch", rows, 512, Format::E4);
  t.middle = allocate(label + " split middle", rows, 2048, Format::E4);
  t.layer0 = allocate(label + " split layer0", rows, 512, Format::E4);
  t.ffnResidual = allocate(label + " split residual", rows, 512, Format::E4);
  t.qkv = allocate(label + " split QKV", rows, 1536, Format::F16);
  t.normalized = allocate(label + " split normalized", rows, 1536, Format::E4);
  t.attended = allocate(label + " split attended", rows, 512, Format::E4);
  return t;
}

void Graph::chainBlock32(Kernels::FusedBlock32Args& f, int block, uint32_t waitScale, bool chainOut) {
  const bool chain = routes_.chain && (routes_.chainMask & 2) && Kernels::fusedBlock32IsPtx(f);
  if (!chain) { c32Prev_.valid = false; return; }
  if (c32Prev_.valid) {
    f.chainWait = c32Prev_.rows; f.chainWaitExpected = c32Prev_.expected; f.chainWaitShiftY = c32Prev_.shiftY; f.chainWaitScale = waitScale;
  }
  if (chainOut) { f.chainSignal = kernels_.syncAddress(block, Kernels::kSyncRows); f.chained = true; }
  c32Prev_.rows = f.chainSignal; c32Prev_.expected = (f.width + f.shiftX + 7) / 8; c32Prev_.shiftY = f.shiftY; c32Prev_.valid = chainOut;
}

// encodeFusedBlock: FFN (dense 32 or C/32 experts) -> QKV -> window attention -> projection.
void Graph::encodeFusedBlock(VkCommandBuffer commands, Temporaries& temps, const Activation& state,
                             Activation* output, int block, uint32_t channels, uint32_t width, uint32_t height,
                             uint32_t phase, const FusedLayout& layout, const Tensor& tensor,
                             const Activation* ffnSkipOverride, Activation* rawOutput, Activation* pooledOutput,
                             uint32_t pooledWidth, bool deferProjection, bool firstInStage, bool lastInStage,
                             uint32_t c32WaitScale, bool c32ChainOut) {
  const uint32_t rows = width * height;
  // Barrier-free chaining of the expert-stage PTX launches (FFN -> attention -> projection / next FFN).
  const bool chain = routes_.chain && (routes_.chainMask & 1) && layout.expertFfn && channels <= 256;
  uint32_t chainShiftX = 0, chainShiftY = 0;
  windowPhase(phase, chainShiftX, chainShiftY);
  const uint32_t chainWindowsX = (width + chainShiftX + 7) / 8;
  const uint32_t chainWinExpected = chainWindowsX * layout.heads;
  // This block's FFN publication alternates between two buffers so a deferred projection can still read the previous one.
  Activation* ffnOut = (temps.ffnQuantized2 && (block & 1)) ? temps.ffnQuantized2 : temps.ffnQuantized;
  kernels_.setStageLabel("block " + std::to_string(block) + " c" + std::to_string(channels));
  const Activation* residual = ffnSkipOverride ? ffnSkipOverride : &state;
  if (options_.fusedBlocks && !options_.captureIntermediates && channels == 32) {
    uint32_t shiftX, shiftY;
    windowPhase(phase, shiftX, shiftY);
    Kernels::FusedBlock32Args f;
    f.state = &state;
    f.skip16 = ffnSkipOverride;
    f.w1 = &model_.fp8Matrix(tensor, layout.expand, 32, layout.hidden);
    f.w2 = &model_.fp8Matrix(tensor, layout.contractWeights, layout.hidden, 32, true, 0, false);   // [32 n][128 k]
    if (Kernels::ptxBlock32Enabled()) {
      static const std::vector<uint32_t> hiddenPerm = Kernels::mlpHiddenPermutation();
      f.w1Ptx = &model_.fp8MatrixPermuted(tensor, layout.expand, 32, layout.hidden, 32, hiddenPerm, "mlp");
      f.w2Ptx = &model_.fp8Matrix(tensor, layout.contractWeights, layout.hidden, 32, true, layout.hidden);   // tile-major [4][32][32]
    }
    f.wqkv = &model_.fp8Matrix(tensor, layout.qkv, 32, 96);
    f.wproj = &model_.fp8Matrix(tensor, layout.projection, 32, 32);
    f.prior = &model_.relativeBias(tensor, layout.relative, 1);
    f.tensor = &tensor;
    f.ffnScaleByteOffset = layout.ffnCosSkip;
    f.attnScaleByteOffset = layout.attnCosSkip;
    f.attentionScaleByteOffset = layout.scale;
    f.outE4 = output;
    f.outF16 = rawOutput;
    f.pooled = pooledOutput; f.pooledWidth = pooledWidth;
    f.width = width; f.height = height; f.shiftX = shiftX; f.shiftY = shiftY;
    chainBlock32(f, block, c32WaitScale, c32ChainOut);
    kernels_.fusedBlock32(commands, f);
    return;
  }
  if (pooledOutput) throw std::runtime_error("pooled output needs the fused 32-channel block");
  uint32_t ffnRowsPerGroup = 64;   // rows per producer workgroup behind the FFN row-band signals (the qkv wait's group size)
  if (layout.expertFfn) {
    // C/32 expert paths, each C -> 128 -> 32 -> C. MpCubicSiLU after W1 only;
    // W3 is expert-major so one K = C GEMM chains every expert into the
    // scaled skip with the F13 accumulation grouping.
    const uint32_t experts = layout.expertCount;
    const uint32_t w2Base = layout.expand + experts * channels * 128;
    const uint32_t w3Base = w2Base + experts * 128 * 32;
    const vk::Buffer& w1Weights = model_.fp8Matrix(tensor, layout.expand, experts * channels, 128, true, channels);
    if (options_.fusedBlocks && !options_.captureIntermediates && Kernels::ptxFfnEnabled() && channels <= 256) {
      // PTX expert FFN + W3 in one kernel (ffn_e4m3.py): every expert of a row group in one workgroup.
      if (ffnSkipOverride) throw std::runtime_error("PTX expert FFN assumes the block state is the FFN skip");
      static const std::vector<uint32_t> hiddenPerm = Kernels::mlpHiddenPermutation();
      Kernels::ExpertFfnArgs ffn;
      ffnRowsPerGroup = 16 * Kernels::ffnRowTiles(channels);
      ffn.input = &state;
      ffn.w1 = &model_.fp8MatrixPermuted(tensor, layout.expand, experts * channels, 128, channels, hiddenPerm, "mlp");
      ffn.w2 = &model_.fp8Matrix(tensor, w2Base, experts * 128, 32, true, 128);
      ffn.w3 = &model_.fp8Matrix(tensor, w3Base, channels, channels);
      ffn.auxTensor = &tensor; ffn.auxByteOffset = layout.ffnCosSkip;
      ffn.output = ffnOut; ffn.rows = rows; ffn.channels = channels; ffn.width = width;
      Kernels::Chain fc;
      if (chain) {
        if (hasPendingProjection_) {
          fc.waitRows = kernels_.syncAddress(pendingProjection_.block, Kernels::kSyncRows);
          fc.waitExpected = pendingProjection_.winExpected; fc.waitShiftY = pendingProjection_.shiftY;
        } else if (!firstInStage) {
          fc.waitBands = kernels_.syncAddress(block, Kernels::kSyncState);   // the previous block's projection GEMM
          fc.waitMul = channels / 64;                                          // one signal per (row group, column group)
        }
        fc.signal = kernels_.syncAddress(block, Kernels::kSyncBands);
        fc.chained = true;
      }
      if (hasPendingProjection_) {   // the previous block's projection runs inside this kernel
        ffn.attended = pendingProjection_.attended; ffn.ffnPrev = pendingProjection_.ffnQ;
        ffn.wproj = pendingProjection_.wproj; ffn.auxAttnByteOffset = pendingProjection_.auxAttn; ffn.auxTensorPrev = pendingProjection_.tensor;
        if (options_.captureBoundaries) ffn.stateOut = pendingProjection_.stateOut;   // materialize the previous block's output for the gate
        hasPendingProjection_ = false;
      }
      kernels_.expertFfnPtx(commands, ffn, fc);
      if (ffn.stateOut) capture(commands, "block-" + std::to_string(pendingProjection_.block), *ffn.stateOut);
    } else if (options_.fusedBlocks && !options_.captureIntermediates) {
      static const bool ptxMlp = !getenv("DLSS5VK_NO_PTX_MLP");
      Kernels::MlpArgs mlp;
      mlp.input = &state; mlp.broadcastInput = true;
      mlp.output = temps.ffnNarrow; mlp.rows = rows; mlp.K = channels; mlp.hidden = 128; mlp.nout = 32;
      mlp.batches = experts;
      if (ptxMlp) {
        // PTX MLP: W1 columns permuted per expert (register-resident hidden layer), W2 tile-major per expert.
        static const std::vector<uint32_t> hiddenPerm = Kernels::mlpHiddenPermutation();
        mlp.w1 = &model_.fp8MatrixPermuted(tensor, layout.expand, experts * channels, 128, channels, hiddenPerm, "mlp");
        mlp.w2 = &model_.fp8Matrix(tensor, w2Base, experts * 128, 32, true, 128);
        kernels_.mlpPtx(commands, mlp);
      } else {
        const vk::Buffer& w2Weights = model_.fp8Matrix(tensor, w2Base, experts * 128, 32, true, 128, false);
        mlp.w1 = &w1Weights; mlp.w2 = &w2Weights;
        kernels_.gemmMlp(commands, mlp);
      }
      GemmFp8Args w3;
      w3.input = temps.ffnNarrow; w3.rows = rows; w3.K = channels; w3.N = channels;
      w3.weights = &model_.fp8Matrix(tensor, w3Base, channels, channels); w3.Nmatrix = channels;
      w3.output = temps.ffnQuantized; w3.quantize = true;   // the f16 twin was never read by the expert blocks
      w3.residual = residual; w3.scaleResidual = true; w3.auxTensor = &tensor; w3.auxByteOffset = layout.ffnCosSkip;
      kernels_.gemmFp8(commands, w3);
    } else {
      const vk::Buffer& w2Weights = model_.fp8Matrix(tensor, w2Base, experts * 128, 32, true, 128);
      GemmFp8Args w1;
      w1.input = &state; w1.rows = rows; w1.K = channels; w1.N = 128; w1.batches = experts; w1.broadcastInput = true;
      w1.weights = &w1Weights; w1.Nmatrix = 128;
      w1.output = temps.ffn; w1.silu = true; w1.quantize = true;
      kernels_.gemmFp8(commands, w1);
      GemmFp8Args w2;
      w2.input = temps.ffn; w2.rows = rows; w2.K = 128; w2.N = 32; w2.batches = experts;
      w2.weights = &w2Weights; w2.Nmatrix = 32;
      w2.output = temps.ffnNarrow; w2.quantize = true;
      kernels_.gemmFp8(commands, w2);
      GemmFp8Args w3;
      w3.input = temps.ffnNarrow; w3.rows = rows; w3.K = channels; w3.N = channels;
      w3.weights = &model_.fp8Matrix(tensor, w3Base, channels, channels); w3.Nmatrix = channels;
      w3.output = temps.ffnResidual; w3.quantize = false; w3.dualOutput = temps.ffnQuantized;
      w3.residual = residual; w3.scaleResidual = true; w3.auxTensor = &tensor; w3.auxByteOffset = layout.ffnCosSkip;
      kernels_.gemmFp8(commands, w3);
    }
  } else {
    GemmFp8Args expand;
    expand.input = &state; expand.rows = rows; expand.K = channels; expand.N = layout.hidden;
    expand.weights = &model_.fp8Matrix(tensor, layout.expand, channels, layout.hidden); expand.Nmatrix = layout.hidden;
    expand.output = temps.ffn; expand.silu = true; expand.quantize = true;
    kernels_.gemmFp8(commands, expand);
    GemmFp8Args contract;
    contract.input = temps.ffn; contract.rows = rows; contract.K = layout.hidden; contract.N = channels;
    contract.weights = &model_.fp8Matrix(tensor, layout.contractWeights, layout.hidden, channels);
    contract.Nmatrix = channels;
    contract.output = temps.ffnResidual; contract.quantize = false; contract.dualOutput = temps.ffnQuantized;
    contract.residual = residual; contract.scaleResidual = true; contract.auxTensor = &tensor;
    contract.auxByteOffset = layout.ffnCosSkip;
    kernels_.gemmFp8(commands, contract);
  }

  std::string prefix = "block-" + std::to_string(block) + "/";
  if (options_.captureIntermediates) {
    capture(commands, prefix + "ffn", *temps.ffn);
    capture(commands, prefix + "ffnResidual", *temps.ffnResidual);
    capture(commands, prefix + "ffnQuantized", *temps.ffnQuantized);
  }
  uint32_t shiftX, shiftY;
  windowPhase(phase, shiftX, shiftY);
  const vk::Buffer& qkvWeights = model_.fp8Matrix(tensor, layout.qkv, channels, channels * 3);
  if (options_.fusedBlocks && !options_.captureIntermediates) {
    // The raw half QKV never leaves the SM: projection + normalize + attention per (window, head).
    Kernels::Chain qc;
    if (chain) {
      qc.waitBands = kernels_.syncAddress(block, Kernels::kSyncBands);
      qc.waitGroupRows = ffnRowsPerGroup;   // the FFN signals each band once per workgroup of this many rows
      qc.signal = kernels_.syncAddress(block, Kernels::kSyncRows);
      qc.chained = true;
    }
    kernels_.qkvAttention(commands, *ffnOut, qkvWeights, channels * 3, tensor, layout.scale,
                          model_.relativeBias(tensor, layout.relative, layout.heads), *temps.attended, width, height,
                          layout.heads, shiftX, shiftY, qc);
  } else {
    GemmFp8Args qkv;
    qkv.input = temps.ffnQuantized; qkv.rows = rows; qkv.K = channels; qkv.N = channels * 3;
    qkv.weights = &qkvWeights; qkv.Nmatrix = channels * 3;
    qkv.output = temps.qkv; qkv.quantize = false;
    kernels_.gemmFp8(commands, qkv);
    if (options_.captureIntermediates) capture(commands, prefix + "qkv", *temps.qkv);
    kernels_.windowNormalize(commands, *temps.qkv, tensor, layout.scale, *temps.normalized, rows, layout.heads);
    if (options_.captureIntermediates) capture(commands, prefix + "normalized", *temps.normalized);
    kernels_.windowAttend(commands, *temps.normalized, model_.relativeBias(tensor, layout.relative, layout.heads),
                          *temps.attended, width, height, layout.heads, shiftX, shiftY);
  }
  if (options_.captureIntermediates) capture(commands, prefix + "attended", *temps.attended);

  // The expert kernels reuse the E4 FFN publication as the attention skip;
  // the dense 32-channel blocks keep the raw FP16 residual.
  if (deferProjection) {
    if (!layout.expertFfn || rawOutput) throw std::runtime_error("deferred projection needs an expert block without a raw output");
    pendingProjection_ = {temps.attended, ffnOut, &model_.fp8Matrix(tensor, layout.projection, channels, channels), layout.attnCosSkip, &tensor, output, block,
                          chainWinExpected, chainShiftY};
    hasPendingProjection_ = true;
    return;
  }
  GemmFp8Args projection;
  projection.input = temps.attended; projection.rows = rows; projection.K = channels; projection.N = channels;
  projection.weights = &model_.fp8Matrix(tensor, layout.projection, channels, channels);
  projection.Nmatrix = channels;
  projection.residual = layout.expertFfn ? (const Activation*)ffnOut : (const Activation*)temps.ffnResidual;
  projection.scaleResidual = true; projection.auxTensor = &tensor; projection.auxByteOffset = layout.attnCosSkip;
  if (rawOutput) {
    projection.output = rawOutput; projection.quantize = false; projection.dualOutput = output;
  } else {
    projection.output = output; projection.quantize = true;
  }
  if (chain) {
    projection.chainWaitRows = kernels_.syncAddress(block, Kernels::kSyncRows);
    projection.chainWaitExpected = chainWinExpected; projection.chainWaitShiftY = chainShiftY; projection.chainWidth = width;
    if (!lastInStage) { projection.chainSignal = kernels_.syncAddress(block + 1, Kernels::kSyncState); projection.chained = true; }
  }
  kernels_.gemmFp8(commands, projection);
}

// encodeSplitBlock: eight independent 512 -> 64 -> 256 -> 64 FFN branches
// (SiLU after the 256-wide middle only), concatenated and contracted, then
// 16-head window attention. Every inter-GEMM boundary is E4M3.
void Graph::encodeSplitBlock(VkCommandBuffer commands, SplitTemporaries& temps, const Activation& state,
                             Activation* output, int block, uint32_t width, uint32_t height, uint32_t phase,
                             Activation* rawOutput, bool firstInStage, bool lastInStage) {
  const uint32_t rows = width * height;
  // Chaining: contract GEMM -> attention -> projection -> next block's layer-0 GEMM without barriers (the branch MLP
  // stays a barrier-separated GLSL dispatch). Bit 4 of the mask is off by default: measured neutral, and leaving it
  // off lets the split GEMMs take the (faster) gemmv route, which chains on a single counter instead.
  const bool chain = routes_.chain && (routes_.chainMask & 4);
  uint32_t chainShiftX = 0, chainShiftY = 0;
  windowPhase(phase, chainShiftX, chainShiftY);
  const uint32_t chainWinExpected = ((width + chainShiftX + 7) / 8) * 16;   // windows x heads
  kernels_.setStageLabel("block " + std::to_string(block) + " split512");
  const uint32_t channels = 512;
  const Tensor& branchTensor = model_.tensor(block, 0);
  const Tensor& contract = model_.tensor(block, 1);
  const Tensor& qkvTensor = model_.tensor(block, 2);
  const Tensor& projection = model_.tensor(block, 3);
  const uint32_t branches = 8, branchChannels = 64, middleChannels = 256;
  const uint32_t w2Base = branches * channels * branchChannels;
  const uint32_t w3Base = w2Base + branches * branchChannels * middleChannels;
  // The qkv tensor of a split block is [512][1536] weights, then the 16 per-head 64x64 priors, then the per-head
  // f32 attention scales; the projection tensor is [512][512] weights then its attention skip scales.
  const uint32_t heads = 16;
  const uint32_t qkvRelative = channels * channels * 3, qkvScale = qkvRelative + heads * 8192;

  GemmFp8Args w1;
  w1.input = &state; w1.rows = rows; w1.K = channels; w1.N = channels;
  w1.weights = &model_.fp8Matrix(branchTensor, 0, channels, channels); w1.Nmatrix = channels;
  w1.output = temps.branch; w1.quantize = true;
  // Chained pieces: 1 contract -> attention, 2 attention -> projection, 4 projection -> the next block's layer 0
  // (the branch MLP stays a barrier-separated dispatch).
  constexpr int splitChain = 7;
  if (chain && !firstInStage && (splitChain & 4)) { w1.chainWaitBands = kernels_.syncAddress(block, Kernels::kSyncState); w1.chainWidth = width; w1.chainWaitMul = channels / 64; }
  kernels_.gemmFp8(commands, w1);
  if (options_.fusedBlocks && !options_.captureIntermediates) {
    const vk::Buffer& w2Weights =
        model_.fp8Matrix(branchTensor, w2Base, branches * branchChannels, middleChannels, true, branchChannels);
    // Fused MLP: W3 N-major per branch ([branch][64 n][256 k]).
    const vk::Buffer& w3Weights =
        model_.fp8Matrix(branchTensor, w3Base, branches * middleChannels, branchChannels, true, middleChannels, false);
    Kernels::MlpArgs mlp;
    mlp.input = temps.branch; mlp.w1 = &w2Weights; mlp.w2 = &w3Weights; mlp.output = temps.layer0;
    mlp.rows = rows; mlp.K = branchChannels; mlp.hidden = middleChannels; mlp.nout = branchChannels; mlp.batches = branches;
    kernels_.gemmMlp(commands, mlp);
  } else {
    const vk::Buffer& w2Weights =
        model_.fp8Matrix(branchTensor, w2Base, branches * branchChannels, middleChannels, true, branchChannels);
    const vk::Buffer& w3Weights =
        model_.fp8Matrix(branchTensor, w3Base, branches * middleChannels, branchChannels, true, middleChannels);
    GemmFp8Args w2;
    w2.input = temps.branch; w2.rows = rows; w2.K = branchChannels; w2.N = middleChannels; w2.batches = branches;
    w2.weights = &w2Weights; w2.Nmatrix = middleChannels; w2.output = temps.middle; w2.silu = true; w2.quantize = true;
    kernels_.gemmFp8(commands, w2);
    GemmFp8Args w3;
    w3.input = temps.middle; w3.rows = rows; w3.K = middleChannels; w3.N = branchChannels; w3.batches = branches;
    w3.weights = &w3Weights; w3.Nmatrix = branchChannels; w3.output = temps.layer0; w3.quantize = true;
    kernels_.gemmFp8(commands, w3);
  }

  GemmFp8Args ffn;
  ffn.input = temps.layer0; ffn.rows = rows; ffn.K = channels; ffn.N = channels;
  ffn.weights = &model_.fp8Matrix(contract, 0, channels, channels); ffn.Nmatrix = channels;
  ffn.output = temps.ffnResidual; ffn.quantize = true;
  ffn.residual = &state; ffn.scaleResidual = true; ffn.auxTensor = &contract; ffn.auxByteOffset = channels * channels;
  if (chain && (splitChain & 1)) { ffn.chainSignal = kernels_.syncAddress(block, Kernels::kSyncBands); ffn.chainWidth = width; ffn.chained = true; }
  kernels_.gemmFp8(commands, ffn);

  uint32_t shiftX, shiftY;
  windowPhase(phase, shiftX, shiftY);
  std::string prefix = "block-" + std::to_string(block) + "/";
  const vk::Buffer& qkvWeights = model_.fp8Matrix(qkvTensor, 0, channels, channels * 3);
  if (options_.fusedBlocks && !options_.captureIntermediates) {
    Kernels::Chain qc;
    if (chain && (splitChain & 1)) { qc.waitBands = kernels_.syncAddress(block, Kernels::kSyncBands); qc.waitMul = channels / 64; }
    if (chain && (splitChain & 2)) { qc.signal = kernels_.syncAddress(block, Kernels::kSyncRows); qc.chained = true; }
    kernels_.qkvAttention(commands, *temps.ffnResidual, qkvWeights, channels * 3, qkvTensor, qkvScale,
                          model_.relativeBias(qkvTensor, qkvRelative, heads), *temps.attended, width, height, heads,
                          shiftX, shiftY, qc);
  } else {
    GemmFp8Args qkv;
    qkv.input = temps.ffnResidual; qkv.rows = rows; qkv.K = channels; qkv.N = channels * 3;
    qkv.weights = &qkvWeights; qkv.Nmatrix = channels * 3;
    qkv.output = temps.qkv; qkv.quantize = false;
    kernels_.gemmFp8(commands, qkv);
    if (options_.captureIntermediates) capture(commands, prefix + "qkv", *temps.qkv);
    kernels_.windowNormalize(commands, *temps.qkv, qkvTensor, qkvScale, *temps.normalized, rows, heads);
    kernels_.windowAttend(commands, *temps.normalized, model_.relativeBias(qkvTensor, qkvRelative, heads),
                          *temps.attended, width, height, heads, shiftX, shiftY);
  }
  if (options_.captureIntermediates) capture(commands, prefix + "attended", *temps.attended);

  GemmFp8Args proj;
  proj.input = temps.attended; proj.rows = rows; proj.K = channels; proj.N = channels;
  proj.weights = &model_.fp8Matrix(projection, 0, channels, channels); proj.Nmatrix = channels;
  proj.residual = temps.ffnResidual; proj.scaleResidual = true; proj.auxTensor = &projection;
  proj.auxByteOffset = channels * channels;
  if (rawOutput) {
    proj.output = rawOutput; proj.quantize = false; proj.dualOutput = output;
  } else {
    proj.output = output; proj.quantize = true;
  }
  if (chain && (splitChain & 2)) {
    proj.chainWaitRows = kernels_.syncAddress(block, Kernels::kSyncRows);
    proj.chainWaitExpected = chainWinExpected; proj.chainWaitShiftY = chainShiftY; proj.chainWidth = width;
  }
  if (chain && (splitChain & 4) && !lastInStage) { proj.chainSignal = kernels_.syncAddress(block + 1, Kernels::kSyncState); proj.chainWidth = width; proj.chained = true; }
  kernels_.gemmFp8(commands, proj);
}

// Global ViT: eight 1024-channel blocks over the coarsest tokens.
void Graph::encodeVit(VkCommandBuffer commands, Activation& state, uint32_t tokens) {
  const uint32_t channels = 1024, heads = 32, ffnChannels = 4096;
  const uint32_t padded = geometry_.paddedVitTokens();
  Activation* expanded = allocate("ViT FFN 4096", tokens, ffnChannels, Format::E4);
  Activation* ffnResidual = allocate("ViT FFN residual", tokens, channels, Format::E4);
  Activation* qkv = allocate("ViT QKV", tokens, channels * 3, Format::F16);
  Activation* normalized = allocate("ViT normalized QKV", padded, channels * 3, Format::E4);
  Activation* attended = allocate("ViT attended", tokens, channels, Format::E4);
  // Counter chaining of the ViT GEMMs (gemmv PTX: one signal per published column group; the consumer waits for
  // the producer's column-group count at kSyncRows + 4 * {0 expand, 1 contract, 2 qkv, 4 projection}).
  const bool chain = routes_.vitChain;
  auto vitCounter = [&](int block, int op) { return kernels_.syncAddress(block, Kernels::kSyncRows) + 4 * op; };
  for (int block = 31; block <= 38; ++block) {
    kernels_.setStageLabel("block " + std::to_string(block) + " vit");
    const Tensor& expand = model_.tensor(block, 0);
    const Tensor& contract = model_.tensor(block, 1);
    const Tensor& qkvTensor = model_.tensor(block, 2);
    const Tensor& projection = model_.tensor(block, 4);
    GemmFp8Args e;
    e.input = &state; e.rows = tokens; e.K = channels; e.N = ffnChannels;
    e.weights = &model_.fp8Matrix(expand, 0, channels, ffnChannels); e.Nmatrix = ffnChannels;
    e.output = expanded; e.silu = true; e.quantize = true;
    e.chainSingle = true;
    if (chain) {
      if (block > 31) { e.chainWaitRows = vitCounter(block - 1, 4); e.chainWaitExpected = Kernels::vitGemmSignals(tokens, channels); }
      e.chainSignal = vitCounter(block, 0); e.chained = true;
    }
    kernels_.gemmFp8(commands, e);
    GemmFp8Args c;
    c.chainSingle = true;
    c.input = expanded; c.rows = tokens; c.K = ffnChannels; c.N = channels; c.partition = 1024;
    c.weights = &model_.fp8Matrix(contract, 0, ffnChannels, channels); c.Nmatrix = channels;
    c.output = ffnResidual; c.quantize = true;
    c.residual = &state; c.scaleResidual = true; c.auxTensor = &contract; c.auxByteOffset = ffnChannels * channels;
    if (chain) { c.chainWaitRows = vitCounter(block, 0); c.chainWaitExpected = Kernels::vitGemmSignals(tokens, ffnChannels); c.chainSignal = vitCounter(block, 1); c.chained = true; }
    kernels_.gemmFp8(commands, c);
    const bool streamPtx = kernels_.globalAttentionStreamPtx(padded) && options_.fusedBlocks;
    const bool attnPtx = (streamPtx || (kernels_.globalAttentionPtx(padded, nullptr) && padded <= 256)) && options_.fusedBlocks;
    GemmFp8Args q;
    q.chainSingle = true;
    q.input = ffnResidual; q.rows = tokens; q.K = channels; q.N = channels * 3; q.partition = 512;
    q.weights = &model_.fp8Matrix(qkvTensor, heads * 4, channels, channels * 3); q.Nmatrix = channels * 3;
    q.output = qkv; q.quantize = false;
    if (chain) {
      q.chainWaitRows = vitCounter(block, 1); q.chainWaitExpected = Kernels::vitGemmSignals(tokens, channels);
      if (attnPtx) { q.chainSignal = vitCounter(block, 2); q.chained = true; }   // else the GLSL attention follows a barrier
    }
    kernels_.gemmFp8(commands, q);
    std::string prefix = "block-" + std::to_string(block) + "/";
    if (options_.captureIntermediates) capture(commands, prefix + "qkv", *qkv);
    if (streamPtx) {
      // any token count: normalize once (per-head E4 layout), then stream the key blocks; both chained
      Kernels::Chain nc, ac;
      if (chain) {
        nc.waitRows = vitCounter(block, 2); nc.waitExpected = Kernels::vitGemmSignals(tokens, channels * 3);
        nc.signal = vitCounter(block, 5); nc.chained = true;
        ac.waitRows = vitCounter(block, 5); ac.waitExpected = heads * (padded / 64); ac.signal = vitCounter(block, 3); ac.chained = true;
      }
      kernels_.globalNormalizePtx(commands, *qkv, qkvTensor, 0, *normalized, tokens, padded, heads, &nc);
      kernels_.globalAttentionStream(commands, *normalized, *attended, tokens, padded, heads, &ac);
    } else if (options_.fusedBlocks && padded <= 256) {
      Kernels::Chain ac;
      if (chain && attnPtx) { ac.waitRows = vitCounter(block, 2); ac.waitExpected = Kernels::vitGemmSignals(tokens, channels * 3); ac.signal = vitCounter(block, 3); ac.chained = true; }
      kernels_.globalAttention(commands, *qkv, qkvTensor, 0, *attended, tokens, padded, heads, nullptr, &ac);
    } else if (options_.fusedBlocks) {
      // Large ViT sequences (high resolutions): normalize once, then stream the keys in chunks.
      kernels_.globalNormalize(commands, *qkv, qkvTensor, 0, *normalized, tokens, heads);
      kernels_.globalAttention(commands, *qkv, qkvTensor, 0, *attended, tokens, padded, heads, normalized);
    } else {
      kernels_.globalNormalize(commands, *qkv, qkvTensor, 0, *normalized, tokens, heads);
      if (options_.captureIntermediates) capture(commands, prefix + "normalized", *normalized);
      kernels_.globalAttend(commands, *normalized, *attended, tokens, padded, heads);
    }
    if (options_.captureIntermediates) capture(commands, prefix + "attended", *attended);
    GemmFp8Args p;
    p.chainSingle = true;
    p.input = attended; p.rows = tokens; p.K = channels; p.N = channels; p.partition = 256;
    p.weights = &model_.fp8Matrix(projection, 0, channels, channels); p.Nmatrix = channels;
    p.output = &state; p.quantize = true;
    p.residual = ffnResidual; p.scaleResidual = true; p.auxTensor = &projection; p.auxByteOffset = channels * channels;
    if (chain && attnPtx) { p.chainWaitRows = vitCounter(block, 3); p.chainWaitExpected = heads * (padded / 64); }
    if (chain && block < 38) { p.chainSignal = vitCounter(block, 4); p.chained = true; }   // the next block's expand waits
    kernels_.gemmFp8(commands, p);
    capture(commands, "block-" + std::to_string(block), state);
  }
}

void Graph::record(VkCommandBuffer commands, const Activation& inputFeatures) {
  kernels_.resetDispatchCount();
  kernels_.resetSync(commands);   // chaining and split-K tile counters
  const Geometry& g = geometry_;
  const uint32_t fullRows = g.fullWidth * g.fullHeight;
  if (inputFeatures.format != Format::F32 || inputFeatures.rows != fullRows || inputFeatures.channels != 16)
    throw std::runtime_error("input features must be f32 [fullWidth*fullHeight][16]");
  boundaries_.clear();
  usedThisRecord_.clear();
  for (uint32_t& phase : windowPhase_) phase = 0;
  usedThisRecord_.insert(inputFeatures.label + "/" + std::to_string(inputFeatures.rows) + "x" + std::to_string(inputFeatures.channels) + "/" + std::to_string((int)inputFeatures.format));

  // ---- Encoder 32 pre: FP16 input adapter, full-resolution block 0, downsample.
  kernels_.setStageLabel("pre");
  const Tensor& preTensor = model_.tensor(0);
  FusedLayout preLayout = preFusedLayout();
  if (preTensor.byteLength != preLayout.endWithoutPadding + 16) throw std::runtime_error("unexpected block0 layout");
  const bool fusePre = routes_.fusePre, fusePool = routes_.fusePool;
  Activation* adapter = allocate("retained full block0", fullRows, 32, Format::E4);
  Activation* resized = allocate("block0 downsample", g.levels[0].width * g.levels[0].height, 32, Format::E4);
  Activation* adapterRaw = allocate("raw FP16 block0", fullRows, 32, Format::F16);
  Temporaries fullTemps = createTemporaries("pre block0", fullRows, 32);
  if (options_.fusedBlocks && !options_.captureIntermediates && fusePre) {
    // The f32 -> f16 conversion and the 16 -> 32 input adapter run inside block 0 (register-resident skip / state).
    uint32_t paddedN = 0;
    const vk::Buffer& adapterWeights = model_.f16Matrix(preTensor, preLayout.inputAdapter, 16, 32, paddedN);
    uint32_t shiftX, shiftY;
    windowPhase(takeWindowPhase(kFullLevel), shiftX, shiftY);
    kernels_.setStageLabel("block 0 c32");
    Kernels::FusedBlock32Args f;
    f.features = &inputFeatures; f.adapterWeights = &adapterWeights;
    f.w1 = &model_.fp8Matrix(preTensor, preLayout.expand, 32, preLayout.hidden);
    f.w2 = &model_.fp8Matrix(preTensor, preLayout.contractWeights, preLayout.hidden, 32, true, 0, false);
    if (Kernels::ptxBlock32Enabled()) {
      static const std::vector<uint32_t> hiddenPerm = Kernels::mlpHiddenPermutation();
      f.w1Ptx = &model_.fp8MatrixPermuted(preTensor, preLayout.expand, 32, preLayout.hidden, 32, hiddenPerm, "mlp");
      f.w2Ptx = &model_.fp8Matrix(preTensor, preLayout.contractWeights, preLayout.hidden, 32, true, preLayout.hidden);   // tile-major [4][32][32]
    }
    f.wqkv = &model_.fp8Matrix(preTensor, preLayout.qkv, 32, 96);
    f.wproj = &model_.fp8Matrix(preTensor, preLayout.projection, 32, 32);
    f.prior = &model_.relativeBias(preTensor, preLayout.relative, 1);
    f.tensor = &preTensor;
    f.ffnScaleByteOffset = preLayout.ffnCosSkip;
    f.attnScaleByteOffset = preLayout.attnCosSkip;
    f.attentionScaleByteOffset = preLayout.scale;
    f.outE4 = adapter;
    if (fusePool) { f.pooled = resized; f.pooledWidth = g.levels[0].width; } else f.outF16 = adapterRaw;
    f.width = g.fullWidth; f.height = g.fullHeight; f.shiftX = shiftX; f.shiftY = shiftY;
    c32Prev_.valid = false;
    chainBlock32(f, 0, 0, fusePool);   // block 1 reads the pooled rows
    kernels_.fusedBlock32(commands, f);
  } else {
    inputHalf_ = allocate("input features f16", fullRows, 16, Format::F16);
    kernels_.convertF32ToF16(commands, inputFeatures, *inputHalf_);
    Activation* projectedFp16 = allocate("full FP16 input adapter", fullRows, 32, Format::F16);
    Activation* projected = allocate("full FP8 input adapter", fullRows, 32, Format::E4);
    {
      uint32_t paddedN = 0;
      GemmF16Args pre;
      pre.input = inputHalf_;
      pre.weights = &model_.f16Matrix(preTensor, preLayout.inputAdapter, 16, 32, paddedN);
      pre.paddedN = paddedN; pre.output = projectedFp16; pre.dualOutput = projected;
      pre.rows = fullRows; pre.K = 16; pre.N = 32;
      kernels_.gemmF16(commands, pre);
    }
    if (options_.captureIntermediates) {
      capture(commands, "block-0/projectedFp16", *projectedFp16);
      capture(commands, "block-0/projected", *projected);
    }
    encodeFusedBlock(commands, fullTemps, *projected, adapter, 0, 32, g.fullWidth, g.fullHeight,
                     takeWindowPhase(kFullLevel), preLayout, preTensor, projectedFp16, adapterRaw);
  }
  capture(commands, "block-0", *adapter);
  if (options_.captureIntermediates) capture(commands, "block-0/adapterRaw", *adapterRaw);

  const Geometry::Level d0 = g.levels[0], d1 = g.levels[1], d2 = g.levels[2], d3 = g.levels[3], d4 = g.levels[4],
                        d5 = g.levels[5];
  const uint32_t rows0 = d0.width * d0.height;
  kernels_.setStageLabel("transition 0-1");
  if (!(options_.fusedBlocks && !options_.captureIntermediates && fusePre && fusePool))
    kernels_.downsample2x(commands, *adapterRaw, *resized, g.fullWidth, g.fullHeight, d0.width, d0.height);
  capture(commands, "transition-0-1", *resized);

  Activation* state = resized;
  Activation* scratch = allocate("encoder 32 state", rows0, 32, Format::E4);
  Activation* transitionRaw32 = allocate("raw FP16 block4", rows0, 32, Format::F16);
  Temporaries latentTemps = createTemporaries("encoder 32", rows0, 32);
  const uint32_t rows1 = d1.width * d1.height;
  Activation* downsampled32 = allocate("encoder 32 downsample", rows1, 32, Format::E4);
  const bool poolBlock4 = options_.fusedBlocks && !options_.captureIntermediates && fusePool;
  for (int block = 1; block <= 4; ++block) {
    encodeFusedBlock(commands, latentTemps, *state, scratch, block, 32, d0.width, d0.height, takeWindowPhase(0),
                     fusedLayout(32), model_.tensor(block), nullptr, (block == 4 && !poolBlock4) ? transitionRaw32 : nullptr,
                     (block == 4 && poolBlock4) ? downsampled32 : nullptr, d1.width, false, true, true,
                     block == 1 ? 1u : 0u, block != 4);
    std::swap(state, scratch);
    capture(commands, "block-" + std::to_string(block), *state);
  }
  Activation* skip32 = state;
  kernels_.setStageLabel("transition 4-5");
  if (!poolBlock4) kernels_.downsample2x(commands, *transitionRaw32, *downsampled32, d0.width, d0.height, d1.width, d1.height);
  capture(commands, "pooled-4-5", *downsampled32);
  Activation* next64 = allocate("encoder 64 input", rows1, 64, Format::E4);
  {
    GemmFp8Args t;
    t.input = downsampled32; t.rows = rows1; t.K = 32; t.N = 64;
    t.weights = &model_.fp8Matrix(model_.tensor(4), fusedLayout(32).endWithoutPadding, 32, 64); t.Nmatrix = 64;
    t.output = next64; t.quantize = true;
    kernels_.gemmFp8(commands, t);
  }
  capture(commands, "transition-4-5", *next64);

  // ---- Encoder fused stages 64 / 128 / 256.
  struct FusedStage { Geometry::Level level, next; uint32_t channels; int first, last, levelIndex; };
  const FusedStage encoderStages[] = {{d1, d2, 64, 5, 8, 1}, {d2, d3, 128, 9, 14, 2}, {d3, d4, 256, 15, 22, 3}};
  Activation* skips[3] = {};
  Activation* stageInput = next64;
  for (int s = 0; s < 3; ++s) {
    const FusedStage& stage = encoderStages[s];
    const uint32_t rows = stage.level.width * stage.level.height;
    const uint32_t nextRows = stage.next.width * stage.next.height;
    std::string label = "encoder " + std::to_string(stage.channels);
    Activation* st = stageInput;
    Activation* sc = allocate(label + " state", rows, stage.channels, Format::E4);
    Activation* raw = allocate(label + " raw transition", rows, stage.channels, Format::F16);
    Temporaries temps = createTemporaries(label, rows, stage.channels);
    const bool deferProj = temps.ffnQuantized2 != nullptr;   // PTX FFN with the projection of the previous block fused
    for (int block = stage.first; block <= stage.last; ++block) {
      encodeFusedBlock(commands, temps, *st, sc, block, stage.channels, stage.level.width, stage.level.height,
                       takeWindowPhase(stage.levelIndex), fusedLayout(stage.channels), model_.tensor(block), nullptr,
                       block == stage.last ? raw : nullptr, nullptr, 0, deferProj && block != stage.last,
                       block == stage.first, block == stage.last);
      std::swap(st, sc);
      if (!hasPendingProjection_) capture(commands, "block-" + std::to_string(block), *st);
    }
    skips[s] = st;
    kernels_.setStageLabel("transition " + std::to_string(stage.last));
    Activation* pooled = allocate(label + " downsample", nextRows, stage.channels, Format::E4);
    kernels_.downsample2x(commands, *raw, *pooled, stage.level.width, stage.level.height, stage.next.width,
                          stage.next.height);
    capture(commands, "pooled-" + std::to_string(stage.last) + "-" + std::to_string(stage.last + 1), *pooled);
    Activation* next = allocate(label + " next stage", nextRows, stage.channels * 2, Format::E4);
    GemmFp8Args t;
    t.input = pooled; t.rows = nextRows; t.K = stage.channels; t.N = stage.channels * 2;
    t.weights = &model_.fp8Matrix(model_.tensor(stage.last), fusedLayout(stage.channels).endWithoutPadding,
                                  stage.channels, stage.channels * 2);
    t.Nmatrix = stage.channels * 2; t.output = next; t.quantize = true;
    kernels_.gemmFp8(commands, t);
    capture(commands, "transition-" + std::to_string(stage.last) + "-" + std::to_string(stage.last + 1), *next);
    stageInput = next;
  }
  Activation* skip64 = skips[0];
  Activation* skip128 = skips[1];
  Activation* skip256 = skips[2];

  // ---- Encoder 512 (split blocks 23-30) and the pooled ViT input.
  const uint32_t rows4 = d4.width * d4.height;
  {
    Activation* st = stageInput;
    Activation* sc = allocate("encoder 512 state", rows4, 512, Format::E4);
    Activation* raw = allocate("encoder 512 raw transition", rows4, 512, Format::F16);
    SplitTemporaries temps = createSplitTemporaries("encoder 512", rows4);
    for (int block = 23; block <= 30; ++block) {
      encodeSplitBlock(commands, temps, *st, sc, block, d4.width, d4.height, takeWindowPhase(4),
                       block == 30 ? raw : nullptr, block == 23, block == 30);
      std::swap(st, sc);
      capture(commands, "block-" + std::to_string(block), *st);
    }
    Activation* skip512 = st;
    kernels_.setStageLabel("transition 30-31");
    const uint32_t tokens = g.vitTokens();
    Activation* pooled = allocate("encoder 512 pooled", tokens, 512, Format::E4);
    kernels_.downsample2x(commands, *raw, *pooled, d4.width, d4.height, d5.width, d5.height);
    Activation* vitState = allocate("ViT state", tokens, 1024, Format::E4);
    GemmFp8Args t;
    t.input = pooled; t.rows = tokens; t.K = 512; t.N = 1024;
    t.weights = &model_.fp8Matrix(model_.tensor(30, 4), 0, 512, 1024); t.Nmatrix = 1024;
    t.output = vitState; t.quantize = true;
    kernels_.gemmFp8(commands, t);

    // ---- ViT 31-38.
    encodeVit(commands, *vitState, tokens);

    // ---- Decoder 512 (39-47): projected ViT output upsampled onto the encoder skip.
    kernels_.setStageLabel("transition 38-39");
    Activation* projected512 = allocate("decoder 512 projection", tokens, 512, Format::F16);
    GemmFp8Args p;
    p.input = vitState; p.rows = tokens; p.K = 1024; p.N = 512; p.partition = 256;
    p.weights = &model_.fp8Matrix(model_.tensor(39), 0, 1024, 512); p.Nmatrix = 512;
    p.output = projected512; p.quantize = false;
    kernels_.gemmFp8(commands, p);
    Activation* merged = allocate("decoder 512 skip merge", rows4, 512, Format::E4);
    kernels_.upsampleResidual(commands, *projected512, *skip512, model_.tensor(39), 1024 * 512, *merged, nullptr,
                              d5.width, d5.height, d4.width, d4.height);
    capture(commands, "block-39", *merged);
    Activation* dst = merged;
    Activation* dsc = allocate("decoder 512 state", rows4, 512, Format::E4);
    SplitTemporaries dtemps = createSplitTemporaries("decoder 512", rows4);
    for (int block = 40; block <= 47; ++block) {
      encodeSplitBlock(commands, dtemps, *dst, dsc, block, d4.width, d4.height, takeWindowPhase(4), nullptr,
                       block == 40, block == 47);
      std::swap(dst, dsc);
      capture(commands, "block-" + std::to_string(block), *dst);
    }
    stageInput = dst;
  }

  // ---- Decoder fused stages 256 / 128 / 64 / 32.
  struct DecoderStage { Geometry::Level low, high; uint32_t channels; int first, last, levelIndex; Activation* skip; };
  const DecoderStage decoderStages[] = {{d4, d3, 256, 48, 55, 3, skip256}, {d3, d2, 128, 56, 61, 2, skip128},
                                        {d2, d1, 64, 62, 65, 1, skip64}, {d1, d0, 32, 66, 69, 0, skip32}};
  for (const DecoderStage& stage : decoderStages) {
    const uint32_t lowRows = stage.low.width * stage.low.height;
    const uint32_t rows = stage.high.width * stage.high.height;
    std::string label = "decoder " + std::to_string(stage.channels);
    const Tensor& transition = model_.tensor(stage.first);
    kernels_.setStageLabel("transition ->" + std::to_string(stage.first));
    FusedLayout layout = upsampleFusedLayout(stage.channels * 2, stage.channels);
    if (transition.byteLength != layout.endWithoutPadding + 16)
      throw std::runtime_error("unexpected upsample layout for block " + std::to_string(stage.first));
    Activation* projection = allocate(label + " projection", lowRows, stage.channels, Format::F16);
    GemmFp8Args p;
    p.input = stageInput; p.rows = lowRows; p.K = stage.channels * 2; p.N = stage.channels;
    p.weights = &model_.fp8Matrix(transition, layout.upsampleWeight, stage.channels * 2, stage.channels);
    p.Nmatrix = stage.channels; p.output = projection; p.quantize = false;
    kernels_.gemmFp8(commands, p);
    Activation* merged = allocate(label + " skip merge", rows, stage.channels, Format::E4);
    Activation* rawMerged = stage.channels == 32 ? allocate(label + " raw skip merge", rows, 32, Format::F16) : nullptr;
    const bool upresInBlock = stage.channels == 32 && options_.fusedBlocks && !options_.captureIntermediates && routes_.fuseUpres;
    if (!upresInBlock)
      kernels_.upsampleResidual(commands, *projection, *stage.skip, transition, layout.transitionScale, *merged,
                                rawMerged, stage.low.width, stage.low.height, stage.high.width, stage.high.height);
    Activation* st = merged;
    Activation* sc = allocate(label + " state", rows, stage.channels, Format::E4);
    Temporaries temps = createTemporaries(label, rows, stage.channels);
    for (int block = stage.first; block <= stage.last; ++block) {
      int index = block - stage.first;
      const uint32_t phase = takeWindowPhase(stage.levelIndex);
      if (index == 0 && upresInBlock) {
        // The 2x upsample + scaled skip merge runs inside the first block's input path (F_UPRES).
        uint32_t shiftX, shiftY;
        windowPhase(phase, shiftX, shiftY);
        kernels_.setStageLabel("block " + std::to_string(block) + " c32");
        const Tensor& tensor = model_.tensor(block);
        Kernels::FusedBlock32Args f;
        f.state = stage.skip; f.lowProjection = projection; f.lowWidth = stage.low.width;
        f.inputScaleByteOffset = layout.transitionScale;
        f.w1 = &model_.fp8Matrix(tensor, layout.expand, 32, layout.hidden);
        f.w2 = &model_.fp8Matrix(tensor, layout.contractWeights, layout.hidden, 32, true, 0, false);
        if (Kernels::ptxBlock32Enabled()) {
          static const std::vector<uint32_t> hiddenPerm = Kernels::mlpHiddenPermutation();
          f.w1Ptx = &model_.fp8MatrixPermuted(tensor, layout.expand, 32, layout.hidden, 32, hiddenPerm, "mlp");
          f.w2Ptx = &model_.fp8Matrix(tensor, layout.contractWeights, layout.hidden, 32, true, layout.hidden);   // tile-major [4][32][32]
        }
        f.wqkv = &model_.fp8Matrix(tensor, layout.qkv, 32, 96);
        f.wproj = &model_.fp8Matrix(tensor, layout.projection, 32, 32);
        f.prior = &model_.relativeBias(tensor, layout.relative, 1);
        f.tensor = &tensor;
        f.ffnScaleByteOffset = layout.ffnCosSkip;
        f.attnScaleByteOffset = layout.attnCosSkip;
        f.attentionScaleByteOffset = layout.scale;
        f.outE4 = sc;
        f.width = stage.high.width; f.height = stage.high.height; f.shiftX = shiftX; f.shiftY = shiftY;
        c32Prev_.valid = false;   // the upsampled inputs come from barrier-separated launches
        chainBlock32(f, block, 0, true);
        kernels_.fusedBlock32(commands, f);
      } else {
        const bool deferProj = temps.ffnQuantized2 != nullptr && block != stage.last;
        encodeFusedBlock(commands, temps, *st, sc, block, stage.channels, stage.high.width, stage.high.height,
                         phase, block == stage.first ? layout : fusedLayout(stage.channels),
                         model_.tensor(block), index == 0 ? rawMerged : nullptr, nullptr, nullptr, 0, deferProj,
                         block == stage.first, block == stage.last, 0u, stage.channels == 32);   // block 69 -> block 70 (post)
      }
      std::swap(st, sc);
      if (!hasPendingProjection_) capture(commands, "block-" + std::to_string(block), *st);
    }
    stageInput = st;
  }

  // ---- Full-resolution post block 70 and the RGBA head.
  {
    const Tensor& tensor = model_.tensor(70);
    kernels_.setStageLabel("post");
    FusedLayout layout = postFusedLayout();
    if (tensor.byteLength != layout.endWithoutPadding) throw std::runtime_error("unexpected block70 layout");
    head_ = allocate("RGBA neural head", fullRows, 4, Format::F32);
    if (options_.fusedBlocks && !options_.captureIntermediates && routes_.fusePost) {
      // The post blend (2x upsample of block 69 + block 0, learned scales) feeds block 70 in registers and the
      // RGBA head runs in its epilogue: the frame's last full-resolution round trips disappear.
      uint32_t paddedN = 0;
      const vk::Buffer& headWeights = model_.f16Matrix(tensor, layout.postWeights, 32, 4, paddedN);
      uint32_t shiftX, shiftY;
      windowPhase(takeWindowPhase(kFullLevel), shiftX, shiftY);
      kernels_.setStageLabel("block 70 c32");
      Kernels::FusedBlock32Args f;
      f.state = adapter; f.lowRes = stageInput; f.lowWidth = d0.width;
      f.inputScaleByteOffset = layout.inputScale; f.adapterScaleByteOffset = layout.adapterScale;
      f.w1 = &model_.fp8Matrix(tensor, layout.expand, 32, layout.hidden);
      f.w2 = &model_.fp8Matrix(tensor, layout.contractWeights, layout.hidden, 32, true, 0, false);
      if (Kernels::ptxBlock32Enabled()) {
        static const std::vector<uint32_t> hiddenPerm = Kernels::mlpHiddenPermutation();
        f.w1Ptx = &model_.fp8MatrixPermuted(tensor, layout.expand, 32, layout.hidden, 32, hiddenPerm, "mlp");
        f.w2Ptx = &model_.fp8Matrix(tensor, layout.contractWeights, layout.hidden, 32, true, layout.hidden);   // tile-major [4][32][32]
      }
      f.wqkv = &model_.fp8Matrix(tensor, layout.qkv, 32, 96);
      f.wproj = &model_.fp8Matrix(tensor, layout.projection, 32, 32);
      f.prior = &model_.relativeBias(tensor, layout.relative, 1);
      f.tensor = &tensor;
      f.ffnScaleByteOffset = layout.ffnCosSkip;
      f.attnScaleByteOffset = layout.attnCosSkip;
      f.attentionScaleByteOffset = layout.scale;
      f.headWeights = &headWeights; f.head = head_;
      f.width = g.fullWidth; f.height = g.fullHeight; f.shiftX = shiftX; f.shiftY = shiftY;
      chainBlock32(f, 70, 2, false);
      kernels_.fusedBlock32(commands, f);
    } else {
      Activation* rawMerged = allocate("post raw merge", fullRows, 32, Format::F16);
      Activation* merged = allocate("post merge", fullRows, 32, Format::E4);
      kernels_.postBlend(commands, *stageInput, *adapter, tensor, layout.inputScale, layout.adapterScale, *rawMerged,
                         *merged, d0.width, d0.height, g.fullWidth, g.fullHeight);
      Activation* rawBlockOutput = allocate("post raw block output", fullRows, 32, Format::F16);
      Temporaries temps = createTemporaries("post", fullRows, 32);
      encodeFusedBlock(commands, temps, *merged, nullptr, 70, 32, g.fullWidth, g.fullHeight,
                       takeWindowPhase(kFullLevel), layout, tensor, rawMerged, rawBlockOutput);
      kernels_.setStageLabel("head");
      uint32_t paddedN = 0;
      GemmF16Args post;
      post.input = rawBlockOutput;
      post.weights = &model_.f16Matrix(tensor, layout.postWeights, 32, 4, paddedN);
      post.paddedN = paddedN; post.output = head_; post.rows = fullRows; post.K = 32; post.N = 4;
      kernels_.gemmF16(commands, post);
    }
  }
  kernels_.checkChainOrder();
}

}  // namespace nr

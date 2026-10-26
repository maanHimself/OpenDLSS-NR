#include "kernels.h"

#include <fstream>
#include <sstream>

#include <cstdlib>
#include <cstring>

namespace nr {

namespace {
constexpr uint32_t F_RESIDUAL = 1, F_SCALE_RESIDUAL = 2, F_SILU = 8, F_QUANTIZE = 16,
                   F_DUAL = 32, F_RESIDUAL_E4 = 64, F_BROADCAST_INPUT = 128, F_OUT_F32 = 256;

// Workgroup shape for the GLSL FP8 GEMM: BM = 16 * subgroups rows, BN = 16 * tiles columns. Take the widest
// column tile that still fills the GPU; below that, the shape that comes closest. Wider tiles beat more
// workgroups (they amortize the staged K64 steps), so "fills" is one and a half workgroups per SM, measured.
// The shape changes no arithmetic: every output element sums its K in the same order whatever the tiling.
void chooseGemmShape(uint32_t N, uint32_t rows, uint32_t batches, uint32_t smCount, uint32_t& subgroups, uint32_t& tiles) {
  if (N % 16) throw std::runtime_error("GEMM N must be a multiple of 16: " + std::to_string(N));
  constexpr uint32_t maxTiles = 8u, maxSubgroups = 4u;
  const uint32_t minGroups = smCount * 3 / 2;
  uint32_t bestGroups = 0;
  subgroups = 4; tiles = 1;
  for (uint32_t sg : {8u, 4u}) {
    if (sg > maxSubgroups) continue;
    uint32_t rowGroups = (rows + 16 * sg - 1) / (16 * sg);
    for (uint32_t t : {8u, 6u, 4u, 3u, 2u, 1u}) {
      if (t > maxTiles || N % (t * 16) != 0) continue;
      uint32_t workgroups = batches * (N / (t * 16)) * rowGroups;
      if (workgroups >= minGroups) { subgroups = sg; tiles = t; return; }
      if (workgroups > bestGroups) { bestGroups = workgroups; subgroups = sg; tiles = t; }
    }
  }
}

void check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

Kernels::Kernels(vk::Context& context, const std::string& shaderDirectory) : context_(context) {
  ptxDirectory_ = getenv("DLSS5VK_PTX_DIR") ? getenv("DLSS5VK_PTX_DIR") : shaderDirectory + "/../ptx";
  for (const char* name : {"gemm_fp8", "gemm_f16", "ops", "window_normalize", "window_attend", "global_normalize",
                           "global_attend", "preprocess", "fused_block32", "qkv_attention", "gemm_mlp", "global_attention", "gemm_reduce"}) {
    modules_[name] = context_.loadShaderModule(shaderDirectory + "/" + name + ".spv");
  }
}

Kernels::~Kernels() {
  for (auto& [key, pipeline] : pipelines_) context_.destroyPipeline(pipeline);
  for (auto& [file, kernel] : ptxKernels_) {
    context_.destroyCudaFunction(kernel.function);
    context_.destroyCudaModule(kernel.module);
  }
  if (profileQueries_) vkDestroyQueryPool(context_.device(), profileQueries_, nullptr);
  context_.destroyBuffer(siluTable_);
  context_.destroyBuffer(syncBuffer_);
  context_.destroyBuffer(chainStatus_);
  context_.destroyBuffer(splitScratch_);
}

void Kernels::setSiluTable(const std::vector<uint16_t>& table) {
  if (table.size() != 65536) throw std::runtime_error("SiLU table must have 65536 entries");
  if (!siluTable_.buffer) siluTable_ = context_.createBuffer(65536 * 2, false, "SiLU table");
  context_.upload(siluTable_, table.data(), 65536 * 2);
}

VkPipeline Kernels::pipeline(const char* shader, const vk::SpecConstants& constants, uint32_t requiredSubgroupSize) {
  std::string key = shader;
  for (uint32_t value : constants.data) key += ":" + std::to_string(value);
  auto it = pipelines_.find(key);
  if (it != pipelines_.end()) return it->second.pipeline;
  vk::Pipeline created = context_.createComputePipeline(modules_.at(shader), constants, shader, requiredSubgroupSize);
  pipelines_[key] = created;
  return created.pipeline;
}

void Kernels::beginProfile(VkCommandBuffer commands, uint32_t maxDispatches) {
  if (profileQueries_) vkDestroyQueryPool(context_.device(), profileQueries_, nullptr);
  profileCapacity_ = maxDispatches + 1;
  profileQueries_ = context_.createTimestampPool(profileCapacity_);
  vkCmdResetQueryPool(commands, profileQueries_, 0, profileCapacity_);
  vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, profileQueries_, 0);
  profileCount_ = 1;
  profileLabels_.clear();
  dispatches_ = 0;
}

std::vector<Kernels::ProfileEntry> Kernels::endProfile() {
  std::vector<ProfileEntry> entries;
  if (!profileQueries_) return entries;
  std::vector<double> stamps = context_.readTimestampsMs(profileQueries_, profileCount_);
  for (uint32_t i = 1; i < profileCount_; ++i) entries.push_back({profileLabels_[i - 1], stamps[i] - stamps[i - 1]});
  vkDestroyQueryPool(context_.device(), profileQueries_, nullptr);
  profileQueries_ = VK_NULL_HANDLE;
  return entries;
}

void Kernels::dispatch(VkCommandBuffer commands, VkPipeline pipeline,
                       const vk::Buffer* const bindings[vk::kGenericBindings], const void* push, uint32_t pushBytes,
                       uint32_t x, uint32_t y, uint32_t z) {
  VkDescriptorSet set = context_.allocateSet(bindings);
  vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, context_.pipelineLayout(), 0, 1, &set, 0, nullptr);
  vkCmdPushConstants(commands, context_.pipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes, push);
  vkCmdDispatch(commands, x, y, z);
  context_.computeBarrier(commands);
  if (profileQueries_ && profileCount_ < profileCapacity_) {
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, profileQueries_, profileCount_++);
    profileLabels_.push_back(stageLabel_ + " | " + dispatchLabel_);
  }
  ++dispatches_;
}

static bool g_chainEnabled = getenv("DLSS5VK_CHAIN") ? atoi(getenv("DLSS5VK_CHAIN")) != 0 : true;
bool Kernels::chainEnabled() { return g_chainEnabled; }
void Kernels::setChainEnabled(bool on) { g_chainEnabled = on; }

VkDeviceAddress Kernels::syncAddress(int block, SyncRegion region) {
  if (syncBuffer_.buffer == VK_NULL_HANDLE) { syncBuffer_ = context_.createBuffer((VkDeviceSize)kSyncSlots * kSyncSlotBytes, false, "sync counters"); context_.fillZero(syncBuffer_); }
  check(block >= 0 && (uint32_t)block < kSyncSlots, "sync slot");
  return context_.deviceAddress(syncBuffer_) + (VkDeviceSize)block * kSyncSlotBytes + (VkDeviceSize)region * kSyncRegionBytes;
}

VkDeviceAddress Kernels::tileCounters(uint32_t count) {
  const uint32_t bytes = (count * 4 + 15) & ~15u;
  check(tileCounterCursor_ + bytes <= (kSyncSlots - kTileCounterSlot) * kSyncSlotBytes, "tile counters exhausted (per frame)");
  VkDeviceAddress address = syncAddress(kTileCounterSlot, kSyncBands) + tileCounterCursor_;
  tileCounterCursor_ += bytes;
  return address;
}

// Consumers spin on these. A stale count starts one early.
void Kernels::resetSync(VkCommandBuffer commands) {
  if (syncBuffer_.buffer == VK_NULL_HANDLE) syncAddress(0, kSyncBands);
  vkCmdFillBuffer(commands, syncBuffer_.buffer, 0, VK_WHOLE_SIZE, 0);
  context_.transferBarrier(commands);
  chainLaunches_.clear();
}

void Kernels::noteChain(VkDeviceAddress waitA, VkDeviceAddress waitB, VkDeviceAddress signal) {
  if (waitA || waitB || signal) chainLaunches_.push_back({stageLabel_ + " | " + dispatchLabel_, {waitA, waitB}, signal});
}

void Kernels::checkChainOrder() const {
  for (size_t i = 0; i < chainLaunches_.size(); ++i) {
    const ChainLaunch& launch = chainLaunches_[i];
    for (VkDeviceAddress wait : launch.waits) {
      if (!wait) continue;
      if (wait == launch.signal) throw std::runtime_error("chain order: " + launch.label + " waits on the counters it signals");
      bool signalledBefore = false;
      for (size_t j = 0; j < i && !signalledBefore; ++j) signalledBefore = chainLaunches_[j].signal == wait;
      if (!signalledBefore) throw std::runtime_error("chain order: " + launch.label + " waits on counters no earlier launch signals");
    }
  }
}

VkDeviceAddress Kernels::chainStatusAddress() {
  if (chainStatus_.buffer == VK_NULL_HANDLE) {
    chainStatus_ = context_.createBuffer(16, true, "chain status");
    memset(chainStatus_.mapped, 0, 16);
  }
  return context_.deviceAddress(chainStatus_);
}

Kernels::ChainTimeouts Kernels::chainTimeouts() const {
  ChainTimeouts timeouts;
  if (!chainStatus_.mapped) return timeouts;
  const volatile uint32_t* status = static_cast<const volatile uint32_t*>(chainStatus_.mapped);
  timeouts.waits = status[0];
  if (timeouts.waits && syncBuffer_.buffer) {
    const uint32_t base = (uint32_t)context_.deviceAddress(syncBuffer_), offset = status[1] - base;
    timeouts.counter = offset < kSyncSlots * kSyncSlotBytes
                           ? "slot " + std::to_string(offset / kSyncSlotBytes) + " region " +
                                 std::to_string(offset % kSyncSlotBytes / kSyncRegionBytes) + " counter " +
                                 std::to_string(offset % kSyncRegionBytes / 4)
                           : "an address outside the counters";
  }
  return timeouts;
}

void Kernels::resetChainTimeouts() {
  if (chainStatus_.mapped) memset(chainStatus_.mapped, 0, 16);
}

void Kernels::cudaLaunchTracked(VkCommandBuffer commands, VkCudaFunctionNV function, uint32_t gridX, uint32_t gridY,
                                uint32_t gridZ, uint32_t blockX, uint32_t sharedBytes, const void* const* params,
                                size_t paramCount, bool chained) {
  context_.cudaLaunch(commands, function, gridX, gridY, gridZ, blockX, sharedBytes, params, paramCount);
  if (!(chained && g_chainEnabled)) context_.computeBarrier(commands);
  if (profileQueries_ && profileCount_ < profileCapacity_) {
    vkCmdWriteTimestamp(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, profileQueries_, profileCount_++);
    profileLabels_.push_back(stageLabel_ + " | " + dispatchLabel_);
  }
  ++dispatches_;
}

void Kernels::dispatchLinear(VkCommandBuffer commands, VkPipeline pipeline,
                             const vk::Buffer* const bindings[vk::kGenericBindings], const void* push,
                             uint32_t pushBytes, uint32_t count) {
  uint32_t groups = (count + 255) / 256;
  uint32_t y = (groups + 65534) / 65535;
  dispatch(commands, pipeline, bindings, push, pushBytes, std::min(groups, 65535u), y, 1);
}

bool Kernels::ptxGemmEnabled() {
  static const bool enabled = getenv("DLSS5VK_PTX_GEMM") ? atoi(getenv("DLSS5VK_PTX_GEMM")) != 0 : true;
  return enabled;
}

bool Kernels::ptxQkvEnabled() {
  static const bool enabled = getenv("DLSS5VK_PTX_QKV") ? atoi(getenv("DLSS5VK_PTX_QKV")) != 0 : true;
  return enabled;
}

void Kernels::gemmFp8(VkCommandBuffer commands, const GemmFp8Args& args) {
  // The scaled residual reads N halves of the aux tensor (every route); the offset is the compacted raw's.
  GemmFp8Args a = args;
  if (a.auxTensor && a.scaleResidual) a.auxByteOffset = a.auxTensor->auxOffset(a.auxByteOffset, 2 * a.N);
  check(a.input && a.input->format == Format::E4, "FP8 GEMM input must be E4M3");
  check(a.K % 32 == 0, "FP8 GEMM K must be a multiple of 32");
  check(a.output, "FP8 GEMM needs an output");
  check(a.output->format == (a.quantize ? Format::E4 : Format::F16), "FP8 GEMM output format mismatch");
  check(!a.dualOutput || (a.dualOutput->format == Format::E4 && !a.quantize), "dual output must be E4 beside f16");
  check(a.output->allocRows >= alignRows(a.rows), "GEMM output rows");
  check(a.input->allocRows >= alignRows(a.rows), "GEMM input rows");
  check(a.inputColumnBase + (a.broadcastInput ? a.K : a.batches * a.K) <= a.input->channels, "GEMM input columns");
  check(a.outputColumnOffset + a.batches * a.N <= a.output->channels, "GEMM output columns");
  check(a.weightColumnOffset + a.N <= a.Nmatrix, "GEMM weight columns");
  check(!a.residual || a.residual->channels == a.output->channels, "residual must share the output layout");
  check(a.partition == 0 || (a.partition % 32 == 0 && a.K % a.partition == 0), "GEMM partition");
  check(a.input->channels % 16 == 0 && a.inputColumnBase % 16 == 0, "GEMM input rows must be 16-byte aligned");
  check(!a.residual || a.residual->allocRows >= alignRows(a.rows), "GEMM residual rows");
  // Split-K: independent partition chains in separate workgroups plus a reduce pass, for the
  // small-M partitioned GEMMs (ViT) whose workgroup count would otherwise starve the GPU.
  uint32_t splits = 1;
  static const int splitEnv = getenv("DLSS5VK_SPLITK") ? atoi(getenv("DLSS5VK_SPLITK")) : -1;   // -1 auto, 0 off, K value: only that K
  if (a.partition && a.K > a.partition && a.batches == 1 &&
      ((a.rows + 63) / 64) * std::max(1u, a.N / 64) < 2 * context_.smCount() && splitEnv != 0 && (splitEnv <= 0 || (uint32_t)splitEnv == a.K) &&
      (VkDeviceSize)(a.K / a.partition) * alignRows(a.rows) * a.N * 2 <= (16u << 20))
    splits = a.K / a.partition;
  // PTX GEMM (scripts/ptx/gemm2_e4m3.py): the expert-stage shapes (single batch, N % 64 == 0, no partition,
  // residual seeded from an E4 tensor with a scale, E4 / f16 / dual outputs, optional SiLU).
  const bool ptxGemm = ptxGemmEnabled();
  // ViT PTX GEMM (gemmv_e4m3.py: 192 x 128 tiles of eight 48 x 64 warps, one chain per workgroup = the whole K or
  // one partition per split-K workgroup, the reduce folded into the last-arriving workgroup, counter chaining).
  const bool ptxGemmV = ptxGemmVEnabled();
  if (ptxGemmV && ptxGemm && a.batches == 1 && !a.broadcastInput && a.N % 128 == 0 &&
      (!a.residual || (a.scaleResidual && a.residual->format == Format::E4)) && a.K / 32 >= 3 &&
      (a.chainSingle || (!a.chainWaitRows && !a.chainWaitBands && !a.chainSignal))) {   // gemmv chains on one counter
    const uint32_t bm = vitGemmTileRows(a.rows), warps = bm / 24;
    const uint32_t rowGroups = (a.rows + bm - 1) / bm, colGroups = a.N / 128;
    // The partition count is the network's K split: the f16 partial sums are combined in that order, so it is part
    // of the arithmetic, not a performance choice. Large token counts just get a larger split-K scratch.
    const uint32_t vsplits = (a.partition && a.K > a.partition) ? a.K / a.partition : 1;
    const uint32_t vflags = (a.residual ? 1u : 0u) | (a.silu ? 2u : 0u) | ((a.quantize || a.dualOutput) ? 4u : 0u) | (a.quantize ? 0u : 8u);
    const std::string entry = "gemmv_e4m3_K" + std::to_string(a.K) + "_f" + std::to_string(vflags) + "_s" + std::to_string(vsplits) +
                              (bm != 192 ? "_m" + std::to_string(bm) : "");
    std::ifstream probe(ptxDirectory_ + "/" + entry + ".ptx");
    const VkDeviceSize partialBytes = (VkDeviceSize)vsplits * rowGroups * colGroups * warps * 48 * 128;   // fragment-order slices
    if (probe.good() && rowGroups <= 65535) {
      PtxKernel& kernel = ptxKernel(entry + ".ptx", entry);
      if (vsplits > 1 && splitScratch_.buffer == VK_NULL_HANDLE) {
        // sized once (launches already recorded hold its address): four times the first split shape covers the
        // widest ViT GEMM of the same token count (qkv: 24 column groups x 2 splits vs the contract's 8 x 4)
        splitScratchBytes_ = std::max<VkDeviceSize>(16u << 20, 4 * partialBytes);
        splitScratch_ = context_.createBuffer(splitScratchBytes_, false, "split-K partials");
      }
      check(vsplits == 1 || partialBytes <= splitScratchBytes_, "split-K partials exceed the scratch buffer (gemmv)");
      VkDeviceAddress pA = context_.deviceAddress(a.input->buffer), pW = context_.deviceAddress(*a.weights);
      VkDeviceAddress pRes = a.residual ? context_.deviceAddress(a.residual->buffer) : pA;
      VkDeviceAddress pAux = a.auxTensor ? context_.deviceAddress(a.auxTensor->raw) : pA;
      VkDeviceAddress pOut = a.quantize ? context_.deviceAddress(a.output->buffer) : a.dualOutput ? context_.deviceAddress(a.dualOutput->buffer) : pA;
      VkDeviceAddress pOut16 = a.quantize ? pA : context_.deviceAddress(a.output->buffer);
      VkDeviceAddress pPartial = vsplits > 1 ? context_.deviceAddress(splitScratch_) : pA;
      VkDeviceAddress pCount = vsplits > 1 ? tileCounters(rowGroups * colGroups) : pA;
      VkDeviceAddress pWait = a.chainWaitRows, pSignal = a.chainSignal;
      uint32_t rows = a.rows, inputStride = a.input->channels, inputColumnBase = a.inputColumnBase, Nmatrix = a.Nmatrix,
               weightColumnOffset = a.weightColumnOffset, outputStride = a.output->channels, outputColumnOffset = a.outputColumnOffset,
               auxHalfOffset = a.auxByteOffset / 2, waitExpected = a.chainWaitExpected;
      VkDeviceAddress pError = chainStatusAddress();
      const void* params[] = {&pA, &pW, &pRes, &pAux, &pOut, &pOut16, &pPartial, &pCount, &rows, &inputStride, &inputColumnBase, &Nmatrix,
                              &weightColumnOffset, &outputStride, &outputColumnOffset, &auxHalfOffset, &pWait, &waitExpected, &pSignal, &pError};
      check(kernel.dynamicShared, "gemmv PTX without a dynamic_shared size");
      dispatchLabel_ = "gemmv_ptx " + std::to_string(a.rows) + "x" + std::to_string(a.K) + "->" + std::to_string(a.N) + " f" +
                       std::to_string(vflags) + (vsplits > 1 ? " s" + std::to_string(vsplits) : "");
      noteChain(pWait, 0, pSignal);
      cudaLaunchTracked(commands, kernel.function, colGroups, rowGroups, vsplits, 32 * warps, kernel.dynamicShared, params, 20, a.chained);
      return;
    }
  }
  // Tall-tile PTX GEMM (gemmt_e4m3.py, 192-row tiles, partitions in the workgroup or one partition per split-K
  // workgroup + a PTX reduce): the few-row stages (ViT / transitions), when the variant was generated.
  static const bool ptxGemmT = getenv("DLSS5VK_PTX_GEMMT") ? atoi(getenv("DLSS5VK_PTX_GEMMT")) != 0 : true;
  if (ptxGemmT && ptxGemm && a.batches == 1 && !a.broadcastInput && a.N % 64 == 0 &&
      (!a.residual || (a.scaleResidual && a.residual->format == Format::E4)) && a.partition != 0 &&   // (rows <= 384 without a partition: gemm2 is faster)
      (a.rows + 191) / 192 <= 65535) {
    const uint32_t partSteps = (a.partition ? a.partition : a.K) / 32;
    const uint32_t rowGroups = (a.rows + 191) / 192, colGroups = a.N / 64;
    uint32_t tsplits = 1;
    if (a.partition && a.K > a.partition && rowGroups * colGroups < 2 * context_.smCount() && splitEnv != 0 &&
        (VkDeviceSize)(a.K / a.partition) * alignRows(a.rows) * a.N * 2 <= (16u << 20))
      tsplits = a.K / a.partition;
    const uint32_t outFlags = (a.silu ? 2u : 0u) | ((a.quantize || a.dualOutput) ? 4u : 0u) | (a.quantize ? 0u : 8u);
    const uint32_t gflags = (a.residual ? 1u : 0u) | (tsplits == 1 ? outFlags : 0u);
    const std::string entry = "gemmt_e4m3_K" + std::to_string(a.K) + "_f" + std::to_string(gflags) + "_p" + std::to_string(partSteps) +
                              "_s" + std::to_string(tsplits);
    std::ifstream probe(ptxDirectory_ + "/" + entry + ".ptx");
    if (probe.good()) {
      check(!a.chainWaitRows && !a.chainWaitBands && !a.chainSignal && !a.chained,
            "the tall-tile PTX GEMM does not implement counter chaining");
      PtxKernel& kernel = ptxKernel(entry + ".ptx", entry);
      const uint32_t splitStride = alignRows(a.rows) * a.N;
      if (tsplits > 1) {
        if (splitScratch_.buffer == VK_NULL_HANDLE) { splitScratchBytes_ = 16u << 20; splitScratch_ = context_.createBuffer(splitScratchBytes_, false, "split-K partials"); }
        check((VkDeviceSize)tsplits * splitStride * 2 <= splitScratchBytes_, "split-K partials exceed the scratch buffer");
      }
      VkDeviceAddress pA = context_.deviceAddress(a.input->buffer), pW = context_.deviceAddress(*a.weights);
      VkDeviceAddress pRes = a.residual ? context_.deviceAddress(a.residual->buffer) : pA;
      VkDeviceAddress pAux = a.auxTensor ? context_.deviceAddress(a.auxTensor->raw) : pA;
      VkDeviceAddress pOut = a.quantize ? context_.deviceAddress(a.output->buffer) : a.dualOutput ? context_.deviceAddress(a.dualOutput->buffer) : pA;
      VkDeviceAddress pOut16 = tsplits > 1 ? context_.deviceAddress(splitScratch_) : a.quantize ? pA : context_.deviceAddress(a.output->buffer);
      uint32_t rows = a.rows, inputStride = a.input->channels, inputColumnBase = a.inputColumnBase, Nmatrix = a.Nmatrix,
               weightColumnOffset = a.weightColumnOffset, outputStride = tsplits > 1 ? a.N : a.output->channels,
               outputColumnOffset = tsplits > 1 ? 0u : a.outputColumnOffset, auxHalfOffset = a.auxByteOffset / 2, sStride = splitStride;
      const void* params[] = {&pA, &pW, &pRes, &pAux, &pOut, &pOut16, &rows, &inputStride, &inputColumnBase, &Nmatrix,
                              &weightColumnOffset, &outputStride, &outputColumnOffset, &auxHalfOffset, &sStride};
      // = gemmt_e4m3.py shared_bytes: a 3-stage ring (8 KB per stage) or the f16 tile, then the E4 tile only next to an f16 tile.
      const bool hasF16 = tsplits > 1 || !a.quantize, hasE4 = tsplits == 1 && (a.quantize || a.dualOutput);
      const uint32_t sharedBytes = std::max(std::min(3u, a.K / 32 / tsplits) * 8192u, hasF16 ? 192u * 128u : 0u) + (hasE4 && hasF16 ? 192u * 64u : 0u);
      dispatchLabel_ = "gemmt_ptx " + std::to_string(a.rows) + "x" + std::to_string(a.K) + "->" + std::to_string(a.N) + " f" +
                       std::to_string(gflags) + (a.partition ? " p" + std::to_string(a.partition) : "") + (tsplits > 1 ? " s" + std::to_string(tsplits) : "");
      cudaLaunchTracked(commands, kernel.function, colGroups, rowGroups, tsplits, 384, sharedBytes, params, 15);
      if (tsplits > 1) {
        const std::string rentry = "reduce_e4m3_s" + std::to_string(tsplits) + "_f" + std::to_string(outFlags);
        PtxKernel& rkernel = ptxKernel(rentry + ".ptx", rentry);
        VkDeviceAddress pPartial = context_.deviceAddress(splitScratch_);
        VkDeviceAddress rOut = a.quantize ? context_.deviceAddress(a.output->buffer) : a.dualOutput ? context_.deviceAddress(a.dualOutput->buffer) : pA;
        VkDeviceAddress rOut16 = a.quantize ? pA : context_.deviceAddress(a.output->buffer);
        uint32_t N = a.N, oStride = a.output->channels, oCol = a.outputColumnOffset;
        const void* rparams[] = {&pPartial, &rOut, &rOut16, &rows, &N, &sStride, &oStride, &oCol};
        dispatchLabel_ = "reduce_ptx " + std::to_string(a.rows) + "x" + std::to_string(a.N) + " s" + std::to_string(tsplits);
        cudaLaunchTracked(commands, rkernel.function, (a.rows * a.N / 8 + 255) / 256, 1, 1, 256, 0, rparams, 8);
      }
      return;
    }
  }
  if (ptxGemm && a.batches == 1 && !a.broadcastInput && a.partition == 0 && a.N % 64 == 0 && splits == 1 &&
      (!a.residual || (a.scaleResidual && a.residual->format == Format::E4)) &&
      (a.rows + 63) / 64 <= 65535) {
    uint32_t pflags = (a.residual ? 1u : 0u) | (a.silu ? 2u : 0u) | ((a.quantize || a.dualOutput) ? 4u : 0u) | (a.quantize ? 0u : 8u);
    const std::string entry = "gemm2_e4m3_K" + std::to_string(a.K) + "_f" + std::to_string(pflags);
    PtxKernel& kernel = ptxKernel(entry + ".ptx", entry);
    VkDeviceAddress pA = context_.deviceAddress(a.input->buffer), pW = context_.deviceAddress(*a.weights);
    VkDeviceAddress pRes = a.residual ? context_.deviceAddress(a.residual->buffer) : pA;
    VkDeviceAddress pAux = a.auxTensor ? context_.deviceAddress(a.auxTensor->raw) : pA;
    VkDeviceAddress pOut = a.quantize ? context_.deviceAddress(a.output->buffer) : a.dualOutput ? context_.deviceAddress(a.dualOutput->buffer) : pA;
    VkDeviceAddress pOut16 = a.quantize ? pA : context_.deviceAddress(a.output->buffer);
    uint32_t rows = a.rows, inputStride = a.input->channels, inputColumnBase = a.inputColumnBase, Nmatrix = a.Nmatrix,
             weightColumnOffset = a.weightColumnOffset, outputStride = a.output->channels, outputColumnOffset = a.outputColumnOffset,
             auxHalfOffset = a.auxByteOffset / 2;
    VkDeviceAddress pWaitRows = a.chainWaitRows, pSignal = a.chainSignal, pWaitBands = a.chainWaitBands;
    uint32_t waitExpected = a.chainWaitExpected, waitShiftY = a.chainWaitShiftY, width = a.chainWidth, waitMul = a.chainWaitMul, waitGroupRows = a.chainWaitGroupRows;
    if ((pWaitRows || pSignal || pWaitBands) && !width)
      throw std::runtime_error("PTX GEMM chaining needs the token row width (" + stageLabel_ + ": " + std::to_string(a.rows) + "x" +
                               std::to_string(a.K) + "->" + std::to_string(a.N) + ")");
    VkDeviceAddress pError = chainStatusAddress();
    const void* params[] = {&pA, &pW, &pRes, &pAux, &pOut, &pOut16, &rows, &inputStride, &inputColumnBase, &Nmatrix,
                            &weightColumnOffset, &outputStride, &outputColumnOffset, &auxHalfOffset,
                            &pWaitRows, &waitExpected, &waitShiftY, &pSignal, &width, &pWaitBands, &waitMul, &waitGroupRows, &pError};
    dispatchLabel_ = "gemm_ptx " + std::to_string(a.rows) + "x" + std::to_string(a.K) + "->" + std::to_string(a.N) + " f" + std::to_string(pflags);
    noteChain(pWaitRows, pWaitBands, pSignal);
    cudaLaunchTracked(commands, kernel.function, a.N / 64, (a.rows + 63) / 64, 1, kernel.threads ? kernel.threads : 128, 0, params, 23, a.chained);
    return;
  }
  if (a.chainWaitRows || a.chainWaitBands || a.chainSignal || a.chained)
    throw std::runtime_error("chained GEMM has no PTX route (" + stageLabel_ + ": " + std::to_string(a.rows) + "x" + std::to_string(a.K) +
                             "->" + std::to_string(a.N) + " p" + std::to_string(a.partition) + ")");
  uint32_t subgroups = 4, tiles = 1;
  chooseGemmShape(a.N, a.rows, a.batches * splits, context_.smCount(), subgroups, tiles);
  uint32_t tile = tiles * 16, blockRows = subgroups * 16;
  uint32_t flags = 0;
  if (a.residual) flags |= F_RESIDUAL;
  if (a.scaleResidual) flags |= F_SCALE_RESIDUAL;
  if (a.silu) flags |= F_SILU;
  if (a.quantize) flags |= F_QUANTIZE;
  if (a.dualOutput) flags |= F_DUAL;
  if (a.residual && a.residual->format == Format::E4) flags |= F_RESIDUAL_E4;
  if (a.broadcastInput) flags |= F_BROADCAST_INPUT;
  check(!a.scaleResidual || a.auxTensor, "scaled residual needs an aux tensor");
  check(a.output->channels % 16 == 0 && (a.outputColumnOffset % 16) == 0 && (a.auxByteOffset % 16) == 0 && a.N % 16 == 0,
        "GEMM epilogue needs 16-column aligned outputs");
  const uint32_t publishFlags = flags & (F_SILU | F_QUANTIZE | F_DUAL);
  if (splits > 1) flags &= ~publishFlags;   // publication moves to the reduce pass
  vk::SpecConstants constants;
  constants.add(0, a.K);
  constants.add(1, tiles);
  constants.add(2, flags);
  constants.add(3, splits > 1 ? 0u : a.partition);
  constexpr uint32_t depthEnv = 4u;   // k32 steps in flight
  constants.add(4, std::max(1u, std::min(depthEnv, std::max(1u, a.K / 32))));
  constants.add(5, 1u);               // padded shared rows
  constants.add(7, subgroups);
  constants.add(8, subgroups * 32);
  constants.add(9, splits);
  // K per staged step: 64 (one flexible 16xNx64 multiply per barrier) when the partition allows it.
  const uint32_t partitionK = a.partition ? a.partition : a.K;
  const uint32_t kstep = partitionK % 64 == 0 ? 64u : 32u;
  constants.add(10, kstep);
  {  // prefetch depth in KSTEP units: the same bytes in flight as DEPTH k32 steps
    const uint32_t stepsTotal = std::max(1u, a.K / kstep / std::max(1u, splits));
    constants.data[4] = std::max(1u, std::min(depthEnv * 32 / kstep, stepsTotal));
  }
  const uint32_t splitStride = alignRows(a.rows) * a.N;   // f16 elements per partition slice
  struct Push {
    uint32_t rows, N, Nmatrix, weightColumnOffset, inputStride, inputColumnBase, outputStride, outputColumnOffset,
        auxHalfOffset, batches, columnGroups, splitStride;
  } push{a.rows, a.N, a.Nmatrix, a.weightColumnOffset, a.input->channels, a.inputColumnBase, a.output->channels,
         a.outputColumnOffset, a.auxByteOffset / 2, a.batches, a.N / tile, splitStride};
  if (splits > 1) {
    // One fixed scratch: it may not be reallocated while recorded dispatches still reference it.
    if (splitScratch_.buffer == VK_NULL_HANDLE) {
      splitScratchBytes_ = 16u << 20;
      splitScratch_ = context_.createBuffer(splitScratchBytes_, false, "split-K partials");
    }
    check((VkDeviceSize)splits * splitStride * 2 <= splitScratchBytes_, "split-K partials exceed the scratch buffer");
  }
  const vk::Buffer* bindings[vk::kGenericBindings] = {};
  bindings[0] = &a.input->buffer;
  bindings[1] = a.weights;
  if (!a.quantize) bindings[2] = &a.output->buffer;
  if (splits > 1) bindings[2] = &splitScratch_;
  if (a.residual && a.residual->format == Format::F16) bindings[3] = &a.residual->buffer;
  if (a.auxTensor) bindings[4] = &a.auxTensor->raw;
  if (a.quantize) bindings[5] = &a.output->buffer;
  if (a.dualOutput) bindings[5] = &a.dualOutput->buffer;
  if (a.residual && a.residual->format == Format::E4) bindings[6] = &a.residual->buffer;
  uint32_t rowGroups = (a.rows + blockRows - 1) / blockRows;
  uint32_t z = (rowGroups + 65534) / 65535;
  dispatchLabel_ = "gemm_fp8 " + std::to_string(a.rows) + "x" + std::to_string(a.K) + "->" + std::to_string(a.N) +
                   (a.batches > 1 ? " x" + std::to_string(a.batches) : "") + (a.silu ? " silu" : "") +
                   (a.partition ? " p" + std::to_string(a.partition) : "") + " f" + std::to_string(flags) +
                   " " + std::to_string(subgroups) + "x" + std::to_string(tiles);
  if (splits > 1) dispatchLabel_ += " s" + std::to_string(splits);
  dispatch(commands, pipeline("gemm_fp8", constants), bindings, &push, sizeof(push), a.batches * (a.N / tile) * splits,
           std::min(rowGroups, 65535u), z);
  if (splits > 1) {
    vk::SpecConstants reduceConstants;
    reduceConstants.add(0, splits);
    reduceConstants.add(1, publishFlags);
    struct ReducePush { uint32_t rows, N, splitStride, outputStride, outputColumnOffset; }
        reducePush{a.rows, a.N, splitStride, a.output->channels, a.outputColumnOffset};
    const vk::Buffer* reduceBindings[vk::kGenericBindings] = {};
    reduceBindings[0] = &splitScratch_;
    if (!a.quantize) reduceBindings[2] = &a.output->buffer;
    if (a.quantize) reduceBindings[5] = &a.output->buffer;
    if (a.dualOutput) reduceBindings[5] = &a.dualOutput->buffer;
    dispatchLabel_ = "gemm_reduce " + std::to_string(a.rows) + "x" + std::to_string(a.N) + " s" + std::to_string(splits);
    dispatchLinear(commands, pipeline("gemm_reduce", reduceConstants), reduceBindings, &reducePush, sizeof(reducePush),
                   a.rows * a.N / 8);
  }
}

void Kernels::gemmF16(VkCommandBuffer commands, const GemmF16Args& a) {
  check(a.input && a.input->format == Format::F16, "f16 GEMM input must be f16");
  check(a.K % 16 == 0 && a.paddedN % 16 == 0 && a.N <= a.paddedN, "f16 GEMM shape");
  check(a.input->allocRows >= alignRows(a.rows) && a.output->allocRows >= alignRows(a.rows), "f16 GEMM rows");
  uint32_t flags = 0;
  if (a.output->format == Format::F32) flags |= F_OUT_F32;
  else if (a.output->format == Format::E4) flags |= F_QUANTIZE;
  if (a.dualOutput) flags |= F_DUAL;
  vk::SpecConstants constants;
  constants.add(0, a.K);
  constants.add(1, a.paddedN);
  constants.add(2, flags);
  struct Push { uint32_t rows, N, inputStride, outputStride; } push{a.rows, a.N, a.input->channels, a.output->channels};
  const vk::Buffer* bindings[vk::kGenericBindings] = {};
  bindings[0] = &a.input->buffer;
  bindings[1] = a.weights;
  if (a.output->format == Format::F16) bindings[2] = &a.output->buffer;
  if (a.output->format == Format::E4) bindings[5] = &a.output->buffer;
  if (a.dualOutput) bindings[5] = &a.dualOutput->buffer;
  if (a.output->format == Format::F32) bindings[7] = &a.output->buffer;
  uint32_t rowGroups = (a.rows + 63) / 64;
  dispatchLabel_ = "gemm_f16 " + std::to_string(a.rows) + "x" + std::to_string(a.K) + "->" + std::to_string(a.N);
  dispatch(commands, pipeline("gemm_f16", constants), bindings, &push, sizeof(push), 1, std::min(rowGroups, 65535u),
           (rowGroups + 65534) / 65535);
}

namespace {
struct OpsPush {
  uint32_t count, channels, inWidth, inHeight, outWidth, outHeight, auxOffsetA, auxOffsetB, dual;
};
}  // namespace

void Kernels::preprocessFromProxy(VkCommandBuffer commands, const vk::Buffer& proxy, Activation& features,
                                  const PreprocessArgs& a) {
  check(features.format == Format::F32 && features.channels == 16 && features.rows == a.fullWidth * a.fullHeight,
        "preprocess output must be f32 [full][16]");
  struct Push {
    uint32_t fullWidth, fullHeight, validWidth, validHeight, sourceWidth, sourceHeight, seed;
    float autoMask, localTone, localStructure, skinStructure, style;
  } push{a.fullWidth, a.fullHeight, a.validWidth, a.validHeight, a.sourceWidth, a.sourceHeight, a.seed,
         a.autoMask ? 1.0f : -1.0f, a.localTone, a.localStructure, a.skinStructure, a.style};
  const vk::Buffer* bindings[vk::kGenericBindings] = {};
  bindings[0] = &proxy;
  bindings[7] = &features.buffer;
  dispatchLabel_ = "preprocess";
  dispatch(commands, pipeline("preprocess", {}, 0), bindings, &push, sizeof(push), (a.fullWidth + 7) / 8,
           (a.fullHeight + 7) / 8, 1);
}

void Kernels::convertF32ToF16(VkCommandBuffer commands, const Activation& input, Activation& output) {
  check(input.format == Format::F32 && output.format == Format::F16, "convert formats");
  uint32_t count = input.rows * input.channels;
  vk::SpecConstants constants;
  constants.add(0, 0);
  OpsPush push{count, input.channels, 0, 0, 0, 0, 0, 0, 0};
  const vk::Buffer* bindings[vk::kGenericBindings] = {};
  bindings[0] = &input.buffer;
  bindings[6] = &output.buffer;
  dispatchLabel_ = "convert f32->f16";
  check(count % 8 == 0, "ops: element count must be a multiple of 8");
  dispatchLinear(commands, pipeline("ops", constants), bindings, &push, sizeof(push), count / 8);
}

void Kernels::quantize(VkCommandBuffer commands, const Activation& input, Activation& output) {
  check(input.format == Format::F16 && output.format == Format::E4, "quantize formats");
  uint32_t count = input.rows * input.channels;
  vk::SpecConstants constants;
  constants.add(0, 1);
  OpsPush push{count, input.channels, 0, 0, 0, 0, 0, 0, 0};
  const vk::Buffer* bindings[vk::kGenericBindings] = {};
  bindings[1] = &input.buffer;
  bindings[5] = &output.buffer;
  dispatchLabel_ = "quantize";
  check(count % 8 == 0, "ops: element count must be a multiple of 8");
  dispatchLinear(commands, pipeline("ops", constants), bindings, &push, sizeof(push), count / 8);
}

void Kernels::downsample2x(VkCommandBuffer commands, const Activation& input, Activation& output, uint32_t inWidth,
                           uint32_t inHeight, uint32_t outWidth, uint32_t outHeight) {
  check(input.format == Format::F16 && output.format == Format::E4, "downsample formats");
  check(input.rows == inWidth * inHeight && output.rows == outWidth * outHeight, "downsample geometry");
  uint32_t count = output.rows * output.channels;
  vk::SpecConstants constants;
  constants.add(0, 2);
  OpsPush push{count, output.channels, inWidth, inHeight, outWidth, outHeight, 0, 0, 0};
  const vk::Buffer* bindings[vk::kGenericBindings] = {};
  bindings[1] = &input.buffer;
  bindings[5] = &output.buffer;
  dispatchLabel_ = "downsample2x";
  check(count % 8 == 0, "ops: element count must be a multiple of 8");
  dispatchLinear(commands, pipeline("ops", constants), bindings, &push, sizeof(push), count / 8);
}

void Kernels::postBlend(VkCommandBuffer commands, const Activation& upsampleSource, const Activation& adapter,
                        const Tensor& tensor, uint32_t inputScaleByteOffset, uint32_t skipScaleByteOffset,
                        Activation& rawOutput, Activation& quantizedOutput, uint32_t inWidth, uint32_t inHeight,
                        uint32_t outWidth, uint32_t outHeight) {
  check(upsampleSource.format == Format::E4 && adapter.format == Format::E4, "post blend inputs");
  check(rawOutput.format == Format::F16 && quantizedOutput.format == Format::E4, "post blend outputs");
  uint32_t count = quantizedOutput.rows * quantizedOutput.channels;
  vk::SpecConstants constants;
  constants.add(0, 3);
  OpsPush push{count, quantizedOutput.channels, inWidth, inHeight, outWidth, outHeight,
               tensor.auxOffset(inputScaleByteOffset, 2 * quantizedOutput.channels) / 2,
               tensor.auxOffset(skipScaleByteOffset, 2 * quantizedOutput.channels) / 2, 1};
  const vk::Buffer* bindings[vk::kGenericBindings] = {};
  bindings[2] = &upsampleSource.buffer;
  bindings[3] = &adapter.buffer;
  bindings[4] = &tensor.raw;
  bindings[5] = &quantizedOutput.buffer;
  bindings[6] = &rawOutput.buffer;
  dispatchLabel_ = "postBlend";
  check(count % 8 == 0, "ops: element count must be a multiple of 8");
  dispatchLinear(commands, pipeline("ops", constants), bindings, &push, sizeof(push), count / 8);
}

void Kernels::upsampleResidual(VkCommandBuffer commands, const Activation& projection, const Activation& skip,
                               const Tensor& tensor, uint32_t scaleByteOffset, Activation& output,
                               Activation* rawOutput, uint32_t inWidth, uint32_t inHeight, uint32_t outWidth,
                               uint32_t outHeight) {
  check(projection.format == Format::F16 && skip.format == Format::E4 && output.format == Format::E4,
        "upsample residual formats");
  check(!rawOutput || rawOutput->format == Format::F16, "upsample residual raw output");
  check(projection.rows == inWidth * inHeight && output.rows == outWidth * outHeight, "upsample residual geometry");
  uint32_t count = output.rows * output.channels;
  vk::SpecConstants constants;
  constants.add(0, 4);
  OpsPush push{count, output.channels, inWidth, inHeight, outWidth, outHeight, tensor.auxOffset(scaleByteOffset, 2 * output.channels) / 2, 0,
               rawOutput ? 1u : 0u};
  const vk::Buffer* bindings[vk::kGenericBindings] = {};
  bindings[1] = &projection.buffer;
  bindings[3] = &skip.buffer;
  bindings[4] = &tensor.raw;
  bindings[5] = &output.buffer;
  if (rawOutput) bindings[6] = &rawOutput->buffer;
  dispatchLabel_ = "upsampleResidual";
  check(count % 8 == 0, "ops: element count must be a multiple of 8");
  dispatchLinear(commands, pipeline("ops", constants), bindings, &push, sizeof(push), count / 8);
}

bool Kernels::ptxGemmVEnabled() {
  static const bool enabled = getenv("DLSS5VK_PTX_GEMMV") ? atoi(getenv("DLSS5VK_PTX_GEMMV")) != 0 : true;
  return enabled && ptxGemmEnabled();   // DLSS5VK_PTX_GEMM=0 turns off every PTX GEMM route
}

uint32_t Kernels::vitGemmTileRows(uint32_t rows) {
  // 192-row tiles (8 warps, one per SM) for the ViT's single row group; 96-row tiles (4 warps, two per SM) for the
  // few-row-group shapes (split-512: 672 rows -> 7 x 4 workgroups, 12% faster than gemm2's 64 x 64 tiles).
  return rows <= 192 ? 192u : 96u;
}

// One signal per published (row group, column group) tile; with split-K the last-arriving split publishes the
// tile, so the count does not depend on the split.
uint32_t Kernels::vitGemmSignals(uint32_t rows, uint32_t N) {
  const uint32_t bm = vitGemmTileRows(rows);
  return (N / 128) * ((rows + bm - 1) / bm);
}

bool Kernels::ptxBlock32Enabled() {
  static const bool enabled = getenv("DLSS5VK_PTX_BLOCK32") ? atoi(getenv("DLSS5VK_PTX_BLOCK32")) != 0 : true;
  return enabled;
}

void Kernels::fusedBlock32Ptx(VkCommandBuffer commands, const FusedBlock32Args& a) {
  check(a.w1Ptx && a.w2Ptx, "PTX fused block needs the permuted / tiled weights");
  check(!a.skip16 && !a.outF16, "PTX fused block: f16 skip / raw output variants are not generated");
  uint32_t flags = (a.outE4 ? 2u : 0u) | (a.features ? 8u : 0u) | (a.lowRes ? 16u : 0u) | (a.head ? 32u : 0u) |
                   (a.pooled ? 64u : 0u) | (a.lowProjection ? 128u : 0u);
  const std::string entry = "block32_e4m3_f" + std::to_string(flags);
  PtxKernel& kernel = ptxKernel(entry + ".ptx", entry);
  uint32_t windowsX = (a.width + a.shiftX + 7) / 8, windowsY = (a.height + a.shiftY + 7) / 8;
  uint32_t windows = windowsX * windowsY;
  VkDeviceAddress pState = context_.deviceAddress(a.features ? a.features->buffer : a.state->buffer);
  VkDeviceAddress pLow = a.lowRes ? context_.deviceAddress(a.lowRes->buffer)
                       : a.lowProjection ? context_.deviceAddress(a.lowProjection->buffer) : pState;
  VkDeviceAddress pW1 = context_.deviceAddress(*a.w1Ptx), pW2 = context_.deviceAddress(*a.w2Ptx),
                  pWqkv = context_.deviceAddress(*a.wqkv), pWproj = context_.deviceAddress(*a.wproj),
                  pAux = context_.deviceAddress(a.tensor->raw), pPrior = context_.deviceAddress(*a.prior);
  VkDeviceAddress pOutE4 = a.outE4 ? context_.deviceAddress(a.outE4->buffer) : pState;
  VkDeviceAddress pOut2 = a.pooled ? context_.deviceAddress(a.pooled->buffer) : a.head ? context_.deviceAddress(a.head->buffer) : pState;
  VkDeviceAddress pWf16 = a.features ? context_.deviceAddress(*a.adapterWeights) : a.head ? context_.deviceAddress(*a.headWeights) : pState;
  uint32_t width = a.width, height = a.height, shiftX = a.shiftX, shiftY = a.shiftY, auxFfnHalf = a.ffnScaleByteOffset / 2,
           auxAttnHalf = a.attnScaleByteOffset / 2, scaleWord = a.attentionScaleByteOffset / 4, auxInputHalf = a.inputScaleByteOffset / 2,
           auxAdapterHalf = a.adapterScaleByteOffset / 2, lowWidth = a.pooled ? a.pooledWidth : a.lowWidth;
  VkDeviceAddress pWait = a.chainWait, pSignal = a.chainSignal;
  uint32_t waitExpected = a.chainWaitExpected, waitShiftY = a.chainWaitShiftY, waitScale = a.chainWaitScale;
  VkDeviceAddress pError = chainStatusAddress();
  const void* params[] = {&pState, &pLow, &pW1, &pW2, &pWqkv, &pWproj, &pAux, &pPrior, &pOutE4, &pOut2, &pWf16,
                          &width, &height, &shiftX, &shiftY, &windowsX, &windows, &auxFfnHalf, &auxAttnHalf, &scaleWord,
                          &auxInputHalf, &auxAdapterHalf, &lowWidth, &pWait, &waitExpected, &waitShiftY, &waitScale, &pSignal, &pError};
  // Persistent: eight resident workgroups per SM, each walking windows. The grid must stay co-resident, which is
  // also what keeps a chained consumer from starving the producer it spins on.
  uint32_t groups = std::min(windows, 8 * context_.smCount());
  dispatchLabel_ = "block32_ptx " + std::to_string(windows) + "w f" + std::to_string(flags);
  noteChain(pWait, 0, pSignal);
  cudaLaunchTracked(commands, kernel.function, groups, 1, 1, 128, 0, params, 29, a.chained);
}

bool Kernels::fusedBlock32IsPtx(const FusedBlock32Args& a) {
  return ptxBlock32Enabled() && a.w1Ptx && a.w2Ptx && !a.skip16 && !a.outF16;
}

void Kernels::fusedBlock32(VkCommandBuffer commands, const FusedBlock32Args& args) {
  // Offsets into the compacted raw: the FFN and attention skip scales (32 halves each) and the one f32 attention
  // scale always, the input blend scales on the upres and post variants, the adapter's on the post one.
  FusedBlock32Args a = args;
  a.ffnScaleByteOffset = a.tensor->auxOffset(a.ffnScaleByteOffset, 64);
  a.attnScaleByteOffset = a.tensor->auxOffset(a.attnScaleByteOffset, 64);
  a.attentionScaleByteOffset = a.tensor->auxOffset(a.attentionScaleByteOffset, 4);
  if (a.lowRes || a.lowProjection) a.inputScaleByteOffset = a.tensor->auxOffset(a.inputScaleByteOffset, 64);
  if (a.lowRes) a.adapterScaleByteOffset = a.tensor->auxOffset(a.adapterScaleByteOffset, 64);
  if (fusedBlock32IsPtx(a)) { fusedBlock32Ptx(commands, a); return; }
  check(!a.chainWait && !a.chainSignal && !a.chained, "the GLSL fused block does not implement counter chaining");
  check(a.features || (a.state && a.state->format == Format::E4 && a.state->channels == 32), "fused block state");
  check(!a.skip16 || (a.skip16->format == Format::F16 && a.skip16->channels == 32), "fused block skip");
  check(a.outE4 || a.outF16 || a.head || a.pooled, "fused block needs an output");
  check(!a.pooled || (a.pooled->format == Format::E4 && a.pooled->channels == 32 && !a.outF16 && !a.head && a.pooledWidth), "fused pool output");
  check(!a.outE4 || (a.outE4->format == Format::E4 && a.outE4->channels == 32), "fused block E4 output");
  check(!a.outF16 || (a.outF16->format == Format::F16 && a.outF16->channels == 32), "fused block f16 output");
  check(a.ffnScaleByteOffset % 16 == 0 && a.attnScaleByteOffset % 16 == 0, "fused block scale vectors must be 16-byte aligned");
  uint32_t flags = (a.skip16 ? 1u : 0u) | (a.outE4 ? 2u : 0u) | (a.outF16 ? 4u : 0u) | (a.features ? 8u : 0u) |
                   (a.lowRes ? 16u : 0u) | (a.head ? 32u : 0u) | (a.pooled ? 64u : 0u) | (a.lowProjection ? 128u : 0u);
  vk::SpecConstants constants;
  constants.add(0, flags);
  uint32_t windowsX = (a.width + a.shiftX + 7) / 8, windowsY = (a.height + a.shiftY + 7) / 8;
  uint32_t windows = windowsX * windowsY;
  struct Push {
    uint32_t width, height, shiftX, shiftY, windowsX, windowCount, groupCount, auxFfnHalfOffset, auxAttnHalfOffset, scaleWordOffset,
        auxInputScaleHalf, auxAdapterScaleHalf, lowWidth;
  } push{a.width, a.height, a.shiftX, a.shiftY, windowsX, windows, 0u, a.ffnScaleByteOffset / 2, a.attnScaleByteOffset / 2,
         a.attentionScaleByteOffset / 4, a.inputScaleByteOffset / 2, a.adapterScaleByteOffset / 2,
         a.pooled ? a.pooledWidth : a.lowWidth};
  const vk::Buffer* bindings[vk::kGenericBindings] = {};
  if (a.state) bindings[0] = &a.state->buffer;
  if (a.skip16) bindings[1] = &a.skip16->buffer;
  bindings[2] = a.w1;
  bindings[3] = a.w2;
  bindings[4] = a.wqkv;
  bindings[5] = a.wproj;
  bindings[6] = &a.tensor->raw;
  bindings[7] = a.prior;
  if (a.outE4) bindings[8] = &a.outE4->buffer;
  if (a.outF16) bindings[9] = &a.outF16->buffer;
  if (a.features) { check(a.features->format == Format::F32 && a.features->channels == 16 && a.adapterWeights, "fused pre inputs"); bindings[0] = &a.features->buffer; bindings[11] = a.adapterWeights; }
  if (a.lowRes) { check(a.lowRes->format == Format::E4 && a.state && a.state->format == Format::E4, "fused post inputs"); bindings[1] = &a.lowRes->buffer; }
  if (a.lowProjection) { check(a.lowProjection->format == Format::F16 && a.lowProjection->channels == 32 && a.state && !a.lowRes && !a.skip16, "fused upres inputs"); bindings[1] = &a.lowProjection->buffer; }
  if (a.head) { check(a.head->format == Format::F32 && a.head->channels == 4 && a.headWeights, "fused head"); bindings[9] = &a.head->buffer; bindings[11] = a.headWeights; }
  if (a.pooled) bindings[9] = &a.pooled->buffer;
  dispatchLabel_ = "fused_block32 " + std::to_string(windows) + "w";
  // Persistent: two resident workgroups per SM, each walking window pairs.
  uint32_t groups = std::min((windows + 1) / 2, 2 * context_.smCount());
  push.groupCount = groups;
  dispatch(commands, pipeline("fused_block32", constants), bindings, &push, sizeof(push), groups, 1, 1);
}

Kernels::PtxKernel& Kernels::ptxKernel(const std::string& file, const std::string& entry) {
  auto it = ptxKernels_.find(file);
  if (it != ptxKernels_.end()) return it->second;
  std::ifstream in(ptxDirectory_ + "/" + file);
  if (!in) throw std::runtime_error("cannot read PTX kernel " + ptxDirectory_ + "/" + file);
  std::stringstream buffer; buffer << in.rdbuf();
  PtxKernel kernel;
  const std::string text = buffer.str();
  if (size_t at = text.find("// dynamic_shared "); at != std::string::npos) kernel.dynamicShared = (uint32_t)atoi(text.c_str() + at + 18);
  if (size_t at = text.find("// threads "); at != std::string::npos) kernel.threads = (uint32_t)atoi(text.c_str() + at + 11);
  kernel.module = context_.createCudaModule(text);
  kernel.function = context_.createCudaFunction(kernel.module, entry.c_str());
  return ptxKernels_[file] = kernel;
}

std::vector<uint32_t> Kernels::mlpHiddenPermutation(uint32_t hidden) {
  // W1' column c holds hidden unit u(c) (mlp_e4m3.py hidden_permutation): per 32-block and 16-half,
  // columns 2t + i -> units 4t + i, columns 8 + 2t + i -> units 4t + 2 + i.
  std::vector<uint32_t> perm(hidden);
  for (uint32_t block = 0; block < hidden / 32; ++block)
    for (uint32_t half = 0; half < 2; ++half)
      for (uint32_t t = 0; t < 4; ++t)
        for (uint32_t i = 0; i < 2; ++i) {
          perm[block * 32 + half * 16 + 2 * t + i] = block * 32 + half * 16 + 4 * t + i;
          perm[block * 32 + half * 16 + 8 + 2 * t + i] = block * 32 + half * 16 + 4 * t + 2 + i;
        }
  return perm;
}

void Kernels::mlpPtx(VkCommandBuffer commands, const MlpArgs& a) {
  check(a.input && a.input->format == Format::E4 && a.output && a.output->format == Format::E4, "PTX MLP formats");
  check(a.hidden % 32 == 0 && a.nout % 16 == 0 && a.K % 32 == 0, "PTX MLP shape");
  check(a.broadcastInput || a.inputColumnBase + a.batches * a.K <= a.input->channels, "PTX MLP input columns");
  check(a.input->channels % 16 == 0 && a.inputColumnBase % 16 == 0 && a.outputColumnOffset % 16 == 0, "PTX MLP alignment");
  check(a.output->allocRows >= alignRows(a.rows) && a.input->allocRows >= alignRows(a.rows), "PTX MLP rows");
  check(a.broadcastInput && a.hidden == 128 && a.nout == 32, "PTX MLP: the expert variant (broadcast input, 128 -> 32) only");
  const std::string entry = "mlp_e4m3_K" + std::to_string(a.K);
  PtxKernel& kernel = ptxKernel(entry + ".ptx", entry);
  VkDeviceAddress aAddr = context_.deviceAddress(a.input->buffer), w1Addr = context_.deviceAddress(*a.w1),
                  w2Addr = context_.deviceAddress(*a.w2), dAddr = context_.deviceAddress(a.output->buffer);
  uint32_t rows = a.rows, inputStride = a.input->channels, inputColumnBase = a.inputColumnBase,
           outputStride = a.output->channels, outputColumnOffset = a.outputColumnOffset;
  const void* params[] = {&aAddr, &w1Addr, &w2Addr, &dAddr, &rows, &inputStride, &inputColumnBase, &outputStride, &outputColumnOffset};
  uint32_t rowGroups = (a.rows + 63) / 64;
  dispatchLabel_ = "mlp_ptx " + std::to_string(a.rows) + "x" + std::to_string(a.K) + "->" + std::to_string(a.hidden) + "->" +
                   std::to_string(a.nout) + " x" + std::to_string(a.batches);
  cudaLaunchTracked(commands, kernel.function, a.batches, rowGroups, 1, 128, 0, params, 9);
}

void Kernels::gemmMlp(VkCommandBuffer commands, const MlpArgs& a) {
  check(a.input && a.input->format == Format::E4, "MLP input must be E4M3");
  check(a.output && a.output->format == Format::E4, "MLP output must be E4M3");
  check(a.K % 32 == 0 && a.hidden % 128 == 0 && a.nout % 16 == 0 && a.nout <= 64, "MLP shape");
  check(a.input->channels % 16 == 0 && a.inputColumnBase % 16 == 0, "MLP input alignment");
  check(a.inputColumnBase + (a.broadcastInput ? a.K : a.batches * a.K) <= a.input->channels, "MLP input columns");
  check(a.outputColumnOffset % 16 == 0 && a.output->channels % 16 == 0 && a.nout % 16 == 0, "MLP output alignment");
  check(a.outputColumnOffset + a.batches * a.nout <= a.output->channels, "MLP output columns");
  check(a.output->allocRows >= alignRows(a.rows) && a.input->allocRows >= alignRows(a.rows), "MLP rows");
  vk::SpecConstants constants;
  constants.add(0, a.K);
  constants.add(1, a.hidden);
  constants.add(2, a.nout);
  constants.add(3, a.broadcastInput ? 1u : 0u);
  constexpr uint32_t mlpDepth = 2u;   // k32 steps in flight
  constants.add(5, std::max(1u, std::min(mlpDepth, a.K / 32)));
  struct Push { uint32_t rows, inputStride, inputColumnBase, outputStride, outputColumnOffset, batches; }
      push{a.rows, a.input->channels, a.inputColumnBase, a.output->channels, a.outputColumnOffset, a.batches};
  const vk::Buffer* bindings[vk::kGenericBindings] = {};
  bindings[0] = &a.input->buffer;
  bindings[1] = a.w1;
  bindings[2] = a.w2;
  bindings[5] = &a.output->buffer;
  uint32_t rowGroups = (a.rows + 63) / 64;
  dispatchLabel_ = "gemm_mlp " + std::to_string(a.rows) + "x" + std::to_string(a.K) + "->" + std::to_string(a.hidden) +
                   "->" + std::to_string(a.nout) + " x" + std::to_string(a.batches);
  dispatch(commands, pipeline("gemm_mlp", constants), bindings, &push, sizeof(push), a.batches,
           std::min(rowGroups, 65535u), (rowGroups + 65534) / 65535);
}

bool Kernels::ptxFfnEnabled() {
  static const bool enabled = getenv("DLSS5VK_PTX_FFN") ? atoi(getenv("DLSS5VK_PTX_FFN")) != 0 : true;
  return enabled;
}

uint32_t Kernels::ffnRowTiles(uint32_t channels) {
  return channels == 256 ? 3u : 4u;   // warps = E * rowTiles (<= 32); C=256: 24 warps at .maxnreg 80 keep one workgroup per SM
}

void Kernels::expertFfnPtx(VkCommandBuffer commands, const ExpertFfnArgs& a, const Chain& chain) {
  check(a.input && a.input->format == Format::E4 && a.output && a.output->format == Format::E4, "PTX FFN formats");
  check(a.channels == 64 || a.channels == 128 || a.channels == 256, "PTX FFN channels");
  check(a.input->channels == a.channels && a.output->channels == a.channels, "PTX FFN strides");
  check(a.output->allocRows >= alignRows(a.rows) && a.input->allocRows >= alignRows(a.rows), "PTX FFN rows");
  const uint32_t E = a.channels / 32;
  const uint32_t rowTiles = ffnRowTiles(a.channels);
  check(rowTiles >= 1 && rowTiles <= 4, "PTX FFN row tiles (the A tile covers 64 rows)");
  const uint32_t sharedBytes = 2u * (E * 4096u + 2048u) + 16u * rowTiles * (a.channels + 16u);   // = ffn_e4m3.py sharedBytes
  const bool proj = a.attended != nullptr;
  check(!proj || (a.ffnPrev && a.wproj && a.auxTensorPrev && a.attended->format == Format::E4 && a.ffnPrev->format == Format::E4 &&
                  a.attended->channels == a.channels && a.ffnPrev->channels == a.channels), "PTX FFN proj inputs");
  const std::string entry = "ffn_e4m3_C" + std::to_string(a.channels) + "_R" + std::to_string(rowTiles) + (proj ? "_proj" : "");
  PtxKernel& kernel = ptxKernel(entry + ".ptx", entry);
  VkDeviceAddress pA = context_.deviceAddress(a.input->buffer), pW1 = context_.deviceAddress(*a.w1),
                  pW2 = context_.deviceAddress(*a.w2), pW3 = context_.deviceAddress(*a.w3),
                  pAux = context_.deviceAddress(a.auxTensor->raw), pOut = context_.deviceAddress(a.output->buffer);
  VkDeviceAddress pAtt = proj ? context_.deviceAddress(a.attended->buffer) : pA;
  VkDeviceAddress pPrev = proj ? context_.deviceAddress(a.ffnPrev->buffer) : pA;
  VkDeviceAddress pWproj = proj ? context_.deviceAddress(*a.wproj) : pA;
  VkDeviceAddress pAuxPrev = proj ? context_.deviceAddress(a.auxTensorPrev->raw) : pAux;
  uint32_t rows = a.rows, auxHalf = a.auxTensor->auxOffset(a.auxByteOffset, 2 * a.channels) / 2,
           auxAttnHalf = proj ? a.auxTensorPrev->auxOffset(a.auxAttnByteOffset, 2 * a.channels) / 2 : 0;
  VkDeviceAddress pStateOut = a.stateOut ? context_.deviceAddress(a.stateOut->buffer) : pA;
  uint32_t storeState = a.stateOut ? 1u : 0u;
  VkDeviceAddress pWaitRows = chain.waitRows, pWaitBands = chain.waitBands, pSignal = chain.signal;
  uint32_t waitExpected = chain.waitExpected, waitShiftY = chain.waitShiftY, width = a.width, waitMul = chain.waitMul, waitGroupRows = chain.waitGroupRows;
  check(!(pWaitRows || pWaitBands || pSignal) || width, "PTX FFN chaining needs the token row width");
  VkDeviceAddress pError = chainStatusAddress();
  const void* params[] = {&pA, &pW1, &pW2, &pW3, &pAux, &pOut, &rows, &auxHalf, &pAtt, &pPrev, &pWproj, &auxAttnHalf, &pAuxPrev,
                          &pStateOut, &storeState, &pWaitRows, &waitExpected, &waitShiftY, &pWaitBands, &pSignal, &width, &waitMul, &waitGroupRows, &pError};
  const uint32_t groups = (a.rows + 16 * rowTiles - 1) / (16 * rowTiles);
  dispatchLabel_ = "ffn_ptx " + std::to_string(a.rows) + "x" + std::to_string(a.channels) + " e" + std::to_string(E) + " r" +
                   std::to_string(rowTiles) + (proj ? " +proj" : "");
  noteChain(pWaitRows, pWaitBands, pSignal);
  cudaLaunchTracked(commands, kernel.function, groups, 1, 1, 32 * E * rowTiles, sharedBytes, params, 24, chain.chained);
}

void Kernels::qkvAttention(VkCommandBuffer commands, const Activation& input, const vk::Buffer& weights, uint32_t Nmatrix,
                           const Tensor& tensor, uint32_t scaleByteOffset, const vk::Buffer& prior, Activation& attended,
                           uint32_t width, uint32_t height, uint32_t heads, uint32_t shiftX, uint32_t shiftY, const Chain& chain) {
  scaleByteOffset = tensor.auxOffset(scaleByteOffset, 4 * heads);   // the per-head f32 scales, in the compacted raw
  check(input.format == Format::E4 && input.channels == heads * 32, "qkv attention input");
  check(Nmatrix == heads * 96, "qkv attention weight columns");
  check(attended.format == Format::E4 && attended.channels == heads * 32, "qkv attention output");
  check(scaleByteOffset % 4 == 0, "qkv attention scale offset");
  uint32_t windowsX = (width + shiftX + 7) / 8, windowsY = (height + shiftY + 7) / 8;
  uint32_t windows = windowsX * windowsY;
  if (ptxQkvEnabled()) {
    // PTX kernel (scripts/ptx/qkv_e4m3.py): grid (heads, windows), 128 threads per (window, head).
    const std::string entry = "qkv_e4m3_K" + std::to_string(heads * 32);
    PtxKernel& kernel = ptxKernel(entry + ".ptx", entry);
    VkDeviceAddress pState = context_.deviceAddress(input.buffer), pW = context_.deviceAddress(weights),
                    pPrior = context_.deviceAddress(prior), pAux = context_.deviceAddress(tensor.raw),
                    pOut = context_.deviceAddress(attended.buffer);
    const uint32_t qkvGroups = 12 * context_.smCount();   // persistent, each walking (window, head) items
    uint32_t w = width, h = height, sx = shiftX, sy = shiftY, wxs = windowsX, scaleWord = scaleByteOffset / 4, wc = windows,
             items = heads * windows;
    VkDeviceAddress pWait = chain.waitBands, pSignal = chain.signal;
    uint32_t waitMul = chain.waitMul, waitGroupRows = chain.waitGroupRows;
    VkDeviceAddress pError = chainStatusAddress();
    const void* params[] = {&pState, &pW, &pPrior, &pAux, &pOut, &w, &h, &sx, &sy, &wxs, &scaleWord, &wc, &items, &pWait, &pSignal, &waitMul, &waitGroupRows, &pError};
    dispatchLabel_ = "qkv_ptx " + std::to_string(windows) + "w x" + std::to_string(heads) + " K" + std::to_string(heads * 32);
    check((heads & (heads - 1)) == 0 && windows * heads < (1u << 24), "PTX qkv item indexing");
    noteChain(pWait, 0, pSignal);
    cudaLaunchTracked(commands, kernel.function, std::min(items, qkvGroups), 1, 1, 128, 0, params, 18, chain.chained);
    return;
  }
  check(!chain.waitBands && !chain.waitRows && !chain.signal && !chain.chained,
        "the GLSL fused QKV + attention does not implement counter chaining");
  struct Push { uint32_t width, height, channels, heads, shiftX, shiftY, windowsX, windowCount, scaleWordOffset, Nmatrix; }
      push{width, height, heads * 32, heads, shiftX, shiftY, windowsX, windows, scaleByteOffset / 4, Nmatrix};
  vk::SpecConstants constants;
  constants.add(0, heads * 32);
  const vk::Buffer* bindings[vk::kGenericBindings] = {};
  bindings[0] = &input.buffer;
  bindings[1] = &weights;
  bindings[2] = &prior;
  bindings[4] = &tensor.raw;
  bindings[5] = &attended.buffer;
  dispatchLabel_ = "qkv_attention " + std::to_string(windows) + "w x" + std::to_string(heads) + " K" + std::to_string(heads * 32);
  dispatch(commands, pipeline("qkv_attention", constants), bindings, &push, sizeof(push), heads,
           std::min(windows, 65535u), (windows + 65534) / 65535);
}

void Kernels::windowNormalize(VkCommandBuffer commands, const Activation& qkv, const Tensor& tensor,
                              uint32_t scaleByteOffset, Activation& normalized, uint32_t tokens, uint32_t heads) {
  scaleByteOffset = tensor.auxOffset(scaleByteOffset, 4 * heads);   // the per-head f32 scales, in the compacted raw
  check(qkv.format == Format::F16 && normalized.format == Format::E4, "normalize formats");
  check(qkv.channels == heads * 96 && normalized.channels == heads * 96, "normalize channels");
  struct Push { uint32_t tokens, heads, channels, scaleWordOffset; } push{tokens, heads, heads * 32, scaleByteOffset / 4};
  const vk::Buffer* bindings[vk::kGenericBindings] = {};
  bindings[1] = &qkv.buffer;
  bindings[4] = &tensor.raw;
  bindings[5] = &normalized.buffer;
  dispatchLabel_ = "window_normalize " + std::to_string(tokens) + "x" + std::to_string(heads);
  dispatchLinear(commands, pipeline("window_normalize", {}), bindings, &push, sizeof(push), tokens * heads);
}

void Kernels::windowAttend(VkCommandBuffer commands, const Activation& normalized, const vk::Buffer& prior,
                           Activation& attended, uint32_t width, uint32_t height, uint32_t heads, uint32_t shiftX,
                           uint32_t shiftY) {
  check(normalized.format == Format::E4 && attended.format == Format::E4, "attend formats");
  check(attended.channels == heads * 32, "attend channels");
  uint32_t windowsX = (width + shiftX + 7) / 8;
  uint32_t windowsY = (height + shiftY + 7) / 8;
  uint32_t windows = windowsX * windowsY;
  struct Push { uint32_t width, height, channels, heads, shiftX, shiftY, windowsX, windowCount; }
      push{width, height, heads * 32, heads, shiftX, shiftY, windowsX, windows};
  const vk::Buffer* bindings[vk::kGenericBindings] = {};
  bindings[0] = &normalized.buffer;
  bindings[1] = &prior;
  bindings[5] = &attended.buffer;
  dispatchLabel_ = "window_attend " + std::to_string(windows) + "w x" + std::to_string(heads);
  dispatch(commands, pipeline("window_attend", {}), bindings, &push, sizeof(push), heads, std::min(windows, 65535u),
           (windows + 65534) / 65535);
}

bool Kernels::ptxGlobalAttentionEnabled() {
  static const bool enabled = getenv("DLSS5VK_PTX_ATTN") ? atoi(getenv("DLSS5VK_PTX_ATTN")) != 0 : true;
  return enabled;
}

bool Kernels::globalAttentionPtx(uint32_t paddedTokens, const Activation* normalized) const {
  return ptxGlobalAttentionEnabled() && !normalized && paddedTokens <= 256;
}

// DLSS5VK_ATTN_STREAM: -1 (default) streamed route for padded > 256 (the resident kernel below), 1 always, 0 never.
bool Kernels::globalAttentionStreamPtx(uint32_t paddedTokens) const {
  static const int mode = getenv("DLSS5VK_ATTN_STREAM") ? atoi(getenv("DLSS5VK_ATTN_STREAM")) : -1;
  if (!ptxGlobalAttentionEnabled() || mode == 0) return false;
  if (mode < 0 && paddedTokens <= 256) return false;
  static const bool available = std::ifstream(ptxDirectory_ + "/global_attention_stream_e4m3.ptx").good() &&
                                std::ifstream(ptxDirectory_ + "/global_normalize_e4m3.ptx").good();
  return available;
}

void Kernels::globalNormalizePtx(VkCommandBuffer commands, const Activation& qkv, const Tensor& tensor, uint32_t scaleByteOffset,
                                 Activation& normalized, uint32_t tokens, uint32_t paddedTokens, uint32_t heads, const Chain* chain) {
  scaleByteOffset = tensor.auxOffset(scaleByteOffset, 4 * heads);   // the per-head f32 scales, in the compacted raw
  check(qkv.format == Format::F16 && qkv.channels == heads * 96 && normalized.format == Format::E4 &&
            (VkDeviceSize)normalized.allocRows * normalized.channels >= (VkDeviceSize)3 * heads * paddedTokens * 32,
        "global normalize (PTX) formats");
  check(paddedTokens % 64 == 0 && paddedTokens >= tokens && scaleByteOffset % 4 == 0, "global normalize (PTX) tokens");
  const std::string entry = "global_normalize_e4m3";
  PtxKernel& kernel = ptxKernel(entry + ".ptx", entry);
  VkDeviceAddress pQkv = context_.deviceAddress(qkv.buffer), pAux = context_.deviceAddress(tensor.raw), pOut = context_.deviceAddress(normalized.buffer);
  VkDeviceAddress pWait = chain ? chain->waitRows : 0, pSignal = chain ? chain->signal : 0;
  uint32_t tok = tokens, pad = paddedTokens, h = heads, scaleWord = scaleByteOffset / 4, waitExpected = chain ? chain->waitExpected : 0;
  VkDeviceAddress pError = chainStatusAddress();
  const void* params[] = {&pQkv, &pAux, &pOut, &tok, &pad, &h, &scaleWord, &pWait, &waitExpected, &pSignal, &pError};
  dispatchLabel_ = "global_normalize_ptx " + std::to_string(tokens) + "t x" + std::to_string(heads);
  noteChain(pWait, 0, pSignal);
  cudaLaunchTracked(commands, kernel.function, paddedTokens / 64, heads, 1, 64, 0, params, 11, chain && chain->chained);
}

void Kernels::globalAttentionStream(VkCommandBuffer commands, const Activation& normalized, Activation& attended, uint32_t tokens,
                                    uint32_t paddedTokens, uint32_t heads, const Chain* chain) {
  check(normalized.format == Format::E4 && attended.format == Format::E4 && attended.channels == heads * 32, "global attention (stream) formats");
  check(paddedTokens % 64 == 0 && paddedTokens >= tokens && attended.allocRows >= tokens, "global attention (stream) tokens");
  const std::string entry = "global_attention_stream_e4m3";
  PtxKernel& kernel = ptxKernel(entry + ".ptx", entry);
  VkDeviceAddress pNorm = context_.deviceAddress(normalized.buffer), pOut = context_.deviceAddress(attended.buffer);
  VkDeviceAddress pWait = chain ? chain->waitRows : 0, pSignal = chain ? chain->signal : 0;
  uint32_t tok = tokens, pad = paddedTokens, h = heads, waitExpected = chain ? chain->waitExpected : 0;
  VkDeviceAddress pError = chainStatusAddress();
  const void* params[] = {&pNorm, &pOut, &tok, &pad, &h, &pWait, &waitExpected, &pSignal, &pError};
  dispatchLabel_ = "global_attention_stream_ptx " + std::to_string(tokens) + "t x" + std::to_string(heads);
  noteChain(pWait, 0, pSignal);
  cudaLaunchTracked(commands, kernel.function, heads, paddedTokens / 64, 1, 128, 0, params, 9, chain && chain->chained);
}

void Kernels::globalAttention(VkCommandBuffer commands, const Activation& qkv, const Tensor& tensor, uint32_t scaleByteOffset,
                              Activation& attended, uint32_t tokens, uint32_t paddedTokens, uint32_t heads,
                              const Activation* normalized, const Chain* chain) {
  scaleByteOffset = tensor.auxOffset(scaleByteOffset, 4 * heads);   // the per-head f32 scales, in the compacted raw
  check(qkv.format == Format::F16 && qkv.channels == heads * 96 && attended.format == Format::E4 &&
            attended.channels == heads * 32, "global attention formats");
  check(paddedTokens % 64 == 0 && paddedTokens >= tokens && qkv.allocRows >= tokens, "global attention tokens");
  check(scaleByteOffset % 4 == 0, "global attention scale offset");
  // PTX route (scripts/ptx/global_attention_e4m3.py): one workgroup per (head, 64-query block), counter chaining.
  if (globalAttentionPtx(paddedTokens, normalized)) {
    const std::string entry = "global_attention_e4m3_p" + std::to_string(paddedTokens);
    PtxKernel& kernel = ptxKernel(entry + ".ptx", entry);
    VkDeviceAddress pQkv = context_.deviceAddress(qkv.buffer), pAux = context_.deviceAddress(tensor.raw), pOut = context_.deviceAddress(attended.buffer);
    VkDeviceAddress pWait = chain ? chain->waitRows : 0, pSignal = chain ? chain->signal : 0;
    uint32_t tok = tokens, h = heads, scaleWord = scaleByteOffset / 4, waitExpected = chain ? chain->waitExpected : 0;
    VkDeviceAddress pError = chainStatusAddress();
    const void* params[] = {&pQkv, &pAux, &pOut, &tok, &h, &scaleWord, &pWait, &waitExpected, &pSignal, &pError};
    check(kernel.dynamicShared, "global attention PTX without a dynamic_shared size");
    dispatchLabel_ = "global_attention_ptx " + std::to_string(tokens) + "t x" + std::to_string(heads);
    noteChain(pWait, 0, pSignal);
    cudaLaunchTracked(commands, kernel.function, heads, paddedTokens / 64, 1, 128, kernel.dynamicShared, params, 10, chain && chain->chained);
    return;
  }
  // Keys are staged in chunks of 256 tokens (16 KB); beyond one chunk the q/k/v are normalized once by
  // global_normalize (an E4 [padded][heads*96] buffer) instead of once per query block.
  const bool prenormalized = normalized != nullptr;
  const uint32_t chunk = std::min(paddedTokens, 256u);
  check(!prenormalized || (normalized->format == Format::E4 && normalized->allocRows >= paddedTokens), "global attention normalized buffer");
  vk::SpecConstants constants;
  constants.add(0, paddedTokens);
  constants.add(1, chunk);
  constants.add(2, prenormalized ? 1u : 0u);
  struct Push { uint32_t tokens, heads, channels, scaleWordOffset; } push{tokens, heads, heads * 32, scaleByteOffset / 4};
  const vk::Buffer* bindings[vk::kGenericBindings] = {};
  bindings[0] = prenormalized ? &normalized->buffer : &qkv.buffer;
  bindings[4] = &tensor.raw;
  bindings[5] = &attended.buffer;
  dispatchLabel_ = "global_attention " + std::to_string(tokens) + "t x" + std::to_string(heads) + (prenormalized ? " pre" : "");
  dispatch(commands, pipeline("global_attention", constants), bindings, &push, sizeof(push), heads, paddedTokens / 64, 1);
}

void Kernels::globalNormalize(VkCommandBuffer commands, const Activation& qkv, const Tensor& tensor,
                              uint32_t scaleByteOffset, Activation& normalized, uint32_t tokens, uint32_t heads) {
  scaleByteOffset = tensor.auxOffset(scaleByteOffset, 4 * heads);   // the per-head f32 scales, in the compacted raw
  check(qkv.format == Format::F16 && normalized.format == Format::E4, "global normalize formats");
  struct Push { uint32_t tokens, heads, channels, scaleWordOffset; } push{tokens, heads, heads * 32, scaleByteOffset / 4};
  const vk::Buffer* bindings[vk::kGenericBindings] = {};
  bindings[1] = &qkv.buffer;
  bindings[4] = &tensor.raw;
  bindings[5] = &normalized.buffer;
  dispatchLabel_ = "global_normalize";
  dispatchLinear(commands, pipeline("global_normalize", {}), bindings, &push, sizeof(push), tokens * heads);
}

void Kernels::globalAttend(VkCommandBuffer commands, const Activation& normalized, Activation& attended,
                           uint32_t tokens, uint32_t paddedTokens, uint32_t heads) {
  check(normalized.format == Format::E4 && attended.format == Format::E4, "global attend formats");
  check(paddedTokens % 64 == 0 && paddedTokens >= tokens && normalized.allocRows >= paddedTokens, "global attend tokens");
  vk::SpecConstants constants;
  constants.add(0, paddedTokens);
  struct Push { uint32_t tokens, heads, channels; } push{tokens, heads, heads * 32};
  const vk::Buffer* bindings[vk::kGenericBindings] = {};
  bindings[0] = &normalized.buffer;
  bindings[5] = &attended.buffer;
  dispatchLabel_ = "global_attend";
  dispatch(commands, pipeline("global_attend", constants), bindings, &push, sizeof(push), heads, paddedTokens / 16, 1);
}

}  // namespace nr

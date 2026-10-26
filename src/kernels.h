// Compute kernel wrappers: pipeline specialization cache plus dispatch helpers with the exact operand contracts
// of the network (GLSL cooperative-matrix reference kernels and the PTX kernels of scripts/ptx/).
#pragma once
#include <map>
#include <string>
#include <vector>

#include "nr_model.h"
#include "vk_context.h"

namespace nr {

enum class Format { E4, F16, F32 };

inline uint32_t formatBytes(Format format) { return format == Format::E4 ? 1 : format == Format::F16 ? 2 : 4; }
inline uint32_t alignRows(uint32_t rows) { return (rows + 63) & ~63u; }

// A [rows][channels] activation tensor. Rows are allocated padded to 64 so
// cooperative-matrix A loads never leave the buffer.
struct Activation {
  vk::Buffer buffer;
  Format format = Format::E4;
  uint32_t rows = 0;
  uint32_t channels = 0;
  uint32_t allocRows = 0;
  std::string label;
  VkDeviceSize validBytes() const { return (VkDeviceSize)rows * channels * formatBytes(format); }
};

struct GemmFp8Args {
  const Activation* input = nullptr;
  uint32_t inputColumnBase = 0;   // added for every batch
  const vk::Buffer* weights = nullptr;  // plain [batches*K][Nmatrix] E4M3
  uint32_t Nmatrix = 0;
  uint32_t weightColumnOffset = 0;
  Activation* output = nullptr;   // F16 unless quantize (E4)
  uint32_t outputColumnOffset = 0;
  Activation* dualOutput = nullptr;  // E4 copy of the published value
  const Activation* residual = nullptr;
  bool scaleResidual = false;
  const Tensor* auxTensor = nullptr;
  uint32_t auxByteOffset = 0;
  bool silu = false;
  bool quantize = true;
  uint32_t rows = 0, K = 0, N = 0;
  uint32_t batches = 1;
  bool broadcastInput = false;
  uint32_t partition = 0;  // 0 or 256/512/1024 (split-K modes)
  // PTX path chaining (see Kernels::Chain): wait on window rows, signal row bands, no barrier after the launch.
  VkDeviceAddress chainWaitRows = 0; uint32_t chainWaitExpected = 0, chainWaitShiftY = 0, chainWidth = 0;
  VkDeviceAddress chainWaitBands = 0; uint32_t chainWaitMul = 1;   // row-band counters of the A / residual producer, its signals per row group
  uint32_t chainWaitGroupRows = 64;                                 // the producer's rows per workgroup (its signals cover whole workgroups)
  VkDeviceAddress chainSignal = 0; bool chained = false;
  bool chainSingle = false;   // chainWaitRows / chainSignal are single completion counters (the ViT GEMM route)
};

struct GemmF16Args {
  const Activation* input = nullptr;  // F16
  const vk::Buffer* weights = nullptr;  // plain [K][paddedN] f16
  uint32_t paddedN = 0;
  Activation* output = nullptr;       // F16 / E4 / F32 by format
  Activation* dualOutput = nullptr;   // E4
  uint32_t rows = 0, K = 0, N = 0;
};

struct KernelChain {
  VkDeviceAddress waitRows = 0; uint32_t waitExpected = 0, waitShiftY = 0;   // window-row counters (attention output)
  VkDeviceAddress waitBands = 0; uint32_t waitMul = 1, waitGroupRows = 64;   // row-band counters (FFN / projection output), signals per row group, the producer's rows per workgroup
  VkDeviceAddress signal = 0;                                                // counters to increment
  bool chained = false;                                                      // no barrier after this launch
};

class Kernels {
 public:
  Kernels(vk::Context& context, const std::string& shaderDirectory);
  ~Kernels();

  void gemmFp8(VkCommandBuffer commands, const GemmFp8Args& args);
  void gemmF16(VkCommandBuffer commands, const GemmF16Args& args);
  static bool ptxGemmEnabled();   // the gemm2 PTX route (DLSS5VK_PTX_GEMM, default on)
  static bool ptxQkvEnabled();    // the fused QKV + window attention PTX route (DLSS5VK_PTX_QKV, default on)

  struct PreprocessArgs {
    uint32_t fullWidth, fullHeight, validWidth, validHeight, sourceWidth, sourceHeight, seed;
    bool autoMask;
    float localTone, localStructure, skinStructure, style;
  };
  // Input features from an RGBA f32 proxy image (display code values).
  void preprocessFromProxy(VkCommandBuffer commands, const vk::Buffer& proxy, Activation& features,
                           const PreprocessArgs& args);
  void convertF32ToF16(VkCommandBuffer commands, const Activation& input, Activation& output);
  void quantize(VkCommandBuffer commands, const Activation& input, Activation& output);
  void downsample2x(VkCommandBuffer commands, const Activation& input, Activation& output, uint32_t inWidth,
                    uint32_t inHeight, uint32_t outWidth, uint32_t outHeight);
  // learned_post_blend: raw f16 and E4 copies of up*inputScale + adapter*skipScale.
  void postBlend(VkCommandBuffer commands, const Activation& upsampleSource, const Activation& adapter,
                 const Tensor& tensor, uint32_t inputScaleByteOffset, uint32_t skipScaleByteOffset,
                 Activation& rawOutput, Activation& quantizedOutput, uint32_t inWidth, uint32_t inHeight,
                 uint32_t outWidth, uint32_t outHeight);
  // Decoder transition: E4 (and optionally raw f16) of round_f16(projection[low] + skip * scale).
  void upsampleResidual(VkCommandBuffer commands, const Activation& projection, const Activation& skip,
                        const Tensor& tensor, uint32_t scaleByteOffset, Activation& output, Activation* rawOutput,
                        uint32_t inWidth, uint32_t inHeight, uint32_t outWidth, uint32_t outHeight);

  // Fused 32-channel Swin block: one dispatch per block (see fused_block32.comp).
  struct FusedBlock32Args {
    // Pre / post fusion (block 0 / block 70): the input adapter, the post blend and the RGBA head in the block.
    const Activation* features = nullptr;   // F_PRE: f32 [tokens][16] input features (replaces state / skip16)
    const vk::Buffer* adapterWeights = nullptr;   // F_PRE: f16 [16][32] (f16Matrix layout 1)
    const Activation* lowRes = nullptr;     // F_POST: half-resolution E4 block output (state = block-0 output)
    const Activation* lowProjection = nullptr;   // F_UPRES: half-resolution f16 projection (state = E4 skip, scaled by inputScale)
    uint32_t inputScaleByteOffset = 0, adapterScaleByteOffset = 0, lowWidth = 0;   // F_POST
    const vk::Buffer* headWeights = nullptr;   // F_HEAD: f16 [32][16] (f16Matrix layout 2, padded)
    Activation* head = nullptr;             // F_HEAD: f32 [tokens][4] output (replaces outF16)
    Activation* pooled = nullptr;           // F_POOL: E4 half-resolution 2x2 box pool of the raw output (replaces outF16)
    uint32_t pooledWidth = 0;
    const Activation* state = nullptr;      // E4 [tokens][32]
    const Activation* skip16 = nullptr;     // optional f16 FFN skip (else the state)
    const vk::Buffer* w1 = nullptr;         // [32][128]
    const vk::Buffer* w2 = nullptr;         // [128][32]
    const vk::Buffer* w1Ptx = nullptr;      // PTX path: W1 with the hidden permutation (mlpHiddenPermutation), tile-major
    const vk::Buffer* w2Ptx = nullptr;      // PTX path: W2 tile-major [4][32][32]
    const vk::Buffer* wqkv = nullptr;       // [32][96]
    const vk::Buffer* wproj = nullptr;      // [32][32]
    const vk::Buffer* prior = nullptr;      // [64][64] f16
    const Tensor* tensor = nullptr;
    uint32_t ffnScaleByteOffset = 0, attnScaleByteOffset = 0, attentionScaleByteOffset = 0;
    Activation* outE4 = nullptr;
    Activation* outF16 = nullptr;
    uint32_t width = 0, height = 0, shiftX = 0, shiftY = 0;
    // chaining (PTX path): wait on the producer block's window rows, signal this block's, no barrier after the launch
    VkDeviceAddress chainWait = 0; uint32_t chainWaitExpected = 0, chainWaitShiftY = 0, chainWaitScale = 0;
    VkDeviceAddress chainSignal = 0; bool chained = false;
  };
  void fusedBlock32(VkCommandBuffer commands, const FusedBlock32Args& args);
  // PTX (VK_NV_cuda_kernel_launch) version of the fused 32-channel block (scripts/ptx/block32_e4m3.py).
  static bool ptxBlock32Enabled();
  // Whether these arguments take the PTX route (only PTX launches take part in counter chaining): the f16 skip and
  // raw-output variants of the block are GLSL only.
  static bool fusedBlock32IsPtx(const FusedBlock32Args& args);
  static bool ptxGemmVEnabled();   // the ViT GEMM PTX route (DLSS5VK_PTX_GEMMV, default on; implies ptxGemmEnabled)
  static uint32_t vitGemmTileRows(uint32_t rows);   // 192 or 96
  static uint32_t vitGemmSignals(uint32_t rows, uint32_t N);   // chain signals of a gemmv launch: one per published tile
  void fusedBlock32Ptx(VkCommandBuffer commands, const FusedBlock32Args& args);

  // Split-K partial sums (one fixed scratch: launches already recorded hold its address).
  vk::Buffer splitScratch_{};
  VkDeviceSize splitScratchBytes_ = 0;

  // Fused two-layer MLP (gemm_mlp.comp / mlp_e4m3.py): E4(E4(SiLU(A W1)) W2) per batch, hidden on chip.
  struct MlpArgs {
    const Activation* input = nullptr;
    uint32_t inputColumnBase = 0;
    bool broadcastInput = false;
    const vk::Buffer* w1 = nullptr;        // tile-major [batch][K/32][hidden][32]
    const vk::Buffer* w2 = nullptr;        // N-major [batch][nout][hidden] (fp8Matrix tileMajor = false)
    Activation* output = nullptr;          // E4, columns outputColumnOffset + batch * nout
    uint32_t outputColumnOffset = 0;
    uint32_t rows = 0, K = 0, hidden = 0, nout = 0, batches = 1;
  };
  void gemmMlp(VkCommandBuffer commands, const MlpArgs& args);
  // PTX (VK_NV_cuda_kernel_launch) version of the expert MLP: mlp_e4m3.py. W1 columns permuted with
  // mlpHiddenPermutation() (register-resident hidden layer), W2 tile-major per expert.
  void mlpPtx(VkCommandBuffer commands, const MlpArgs& args);
  static std::vector<uint32_t> mlpHiddenPermutation(uint32_t hidden = 128);
  struct PtxKernel { VkCudaModuleNV module = VK_NULL_HANDLE; VkCudaFunctionNV function = VK_NULL_HANDLE; uint32_t dynamicShared = 0, threads = 0; };
  std::map<std::string, PtxKernel> ptxKernels_;
  PtxKernel& ptxKernel(const std::string& file, const std::string& entry);
  std::string ptxDirectory_;

  // Barrier-free chaining of consecutive PTX launches: producers increment per-frame dependency counters after their
  // stores, consumers spin on them (see swin.py sync_wait / sync_signal). Counter slots per block in syncBuffer_:
  // row bands of the FFN output (kSyncBands), window rows of the attention output (kSyncRows), row bands of the
  // projection output = the next block's state (kSyncState). A launch marked `chained` omits the compute barrier.
  static bool chainEnabled();
  static void setChainEnabled(bool on);
  enum SyncRegion { kSyncBands = 0, kSyncRows = 1, kSyncState = 2 };
  // Counters within a region are indexed by pixel-row band (row / 8) or by window row, so the region bounds the
  // field height a chain can cover.
  static constexpr uint32_t kSyncRegionBytes = 2048, kSyncCountersPerRegion = kSyncRegionBytes / 4;
  VkDeviceAddress syncAddress(int block, SyncRegion region);
  void resetSync(VkCommandBuffer commands);   // zero the counters (top of every frame)
  // Every counter wait of a recording must be on counters an earlier launch of the same recording signals, and on
  // none the waiting launch signals itself: the chain is a forward DAG. Graph::record checks it every time.
  void checkChainOrder() const;
  // A wait that outlives swin.WAIT_LIMIT_NS gives up instead of hanging (the frame completes with wrong bytes) and
  // records itself in a host-visible word that only resetChainTimeouts() clears; read it once the frame is done.
  struct ChainTimeouts { uint32_t waits = 0; std::string counter; };   // counter: where the first stuck wait was
  ChainTimeouts chainTimeouts() const;
  void resetChainTimeouts();
  using Chain = KernelChain;   // namespace scope: GCC rejects Chain() default arguments for a nested struct with member initializers
  struct ExpertFfnArgs {
    const Activation* input = nullptr;     // block state, E4 [rows][channels]; also the skip
    const vk::Buffer* w1 = nullptr;        // tile-major [expert][channels/32][128][32]
    const vk::Buffer* w2 = nullptr;        // N-major [expert][32][128]
    const vk::Buffer* w3 = nullptr;        // tile-major [channels/32][channels][32]
    const Tensor* auxTensor = nullptr;     // ffn skip scale halves at auxByteOffset
    uint32_t auxByteOffset = 0;
    Activation* output = nullptr;          // E4 [rows][channels]
    uint32_t rows = 0, channels = 0;
    // PTX proj fusion: the previous block's projection (attended x wproj + ffnPrev * attn scale) computed in the
    // kernel as the block state (input == nullptr use); attended / ffnPrev E4 [rows][channels].
    const Activation* attended = nullptr;
    const Activation* ffnPrev = nullptr;
    const vk::Buffer* wproj = nullptr;     // tile-major [channels/32][channels][32]
    const Tensor* auxTensorPrev = nullptr; // the previous block's tensor (its attention skip scales)
    uint32_t auxAttnByteOffset = 0;
    Activation* stateOut = nullptr;        // optional: also write the computed block state (parity captures)
    uint32_t width = 0;                    // tokens per pixel row (chaining row-band math)
  };
  // Expert FFN + W3 in one PTX kernel (scripts/ptx/ffn_e4m3.py): w1 permuted tile-major [e][K/32][128][32],
  // w2 tile-major [e][4][32][32], w3 tile-major [K/32][K][32].
  static bool ptxFfnEnabled();
  static uint32_t ffnRowTiles(uint32_t channels);          // PTX expert FFN: 16-row tiles per workgroup (its chaining signals cover 16 * tiles rows)
  void expertFfnPtx(VkCommandBuffer commands, const ExpertFfnArgs& args, const Chain& chain = Chain());

  // Fused QKV projection + normalize + attention (qkv_attention.comp): E4 input in, E4 attended out.
  void qkvAttention(VkCommandBuffer commands, const Activation& input, const vk::Buffer& weights, uint32_t Nmatrix,
                    const Tensor& tensor, uint32_t scaleByteOffset, const vk::Buffer& prior, Activation& attended,
                    uint32_t width, uint32_t height, uint32_t heads, uint32_t shiftX, uint32_t shiftY,
                    const Chain& chain = Chain());
  void windowNormalize(VkCommandBuffer commands, const Activation& qkv, const Tensor& tensor,
                       uint32_t scaleByteOffset, Activation& normalized, uint32_t tokens, uint32_t heads);
  void windowAttend(VkCommandBuffer commands, const Activation& normalized, const vk::Buffer& prior,
                    Activation& attended, uint32_t width, uint32_t height, uint32_t heads, uint32_t shiftX,
                    uint32_t shiftY);
  // Fused ViT normalize + global attention (global_attention.comp): raw half QKV in, E4 attended out.
  void globalAttention(VkCommandBuffer commands, const Activation& qkv, const Tensor& tensor, uint32_t scaleByteOffset,
                       Activation& attended, uint32_t tokens, uint32_t paddedTokens, uint32_t heads,
                       const Activation* normalized = nullptr, const Chain* chain = nullptr);   // normalized: E4 q/k/v from globalNormalize (large token counts)
  static bool ptxGlobalAttentionEnabled();
  bool globalAttentionPtx(uint32_t paddedTokens, const Activation* normalized) const;   // the PTX route applies
  // Streamed PTX route for any token count (global_attention_stream_e4m3.py): normalize once per (token, head)
  // into the per-head E4 layout, then attention with the key blocks streamed through shared memory.
  bool globalAttentionStreamPtx(uint32_t paddedTokens) const;
  void globalNormalizePtx(VkCommandBuffer commands, const Activation& qkv, const Tensor& tensor, uint32_t scaleByteOffset,
                          Activation& normalized, uint32_t tokens, uint32_t paddedTokens, uint32_t heads, const Chain* chain);
  void globalAttentionStream(VkCommandBuffer commands, const Activation& normalized, Activation& attended, uint32_t tokens,
                             uint32_t paddedTokens, uint32_t heads, const Chain* chain);
  void globalNormalize(VkCommandBuffer commands, const Activation& qkv, const Tensor& tensor,
                       uint32_t scaleByteOffset, Activation& normalized, uint32_t tokens, uint32_t heads);
  void globalAttend(VkCommandBuffer commands, const Activation& normalized, Activation& attended, uint32_t tokens,
                    uint32_t paddedTokens, uint32_t heads);

  uint32_t dispatchCount() const { return dispatches_; }
  void resetDispatchCount() { dispatches_ = 0; tileCounterCursor_ = 0; }
  // Per-frame zeroed tile counters (split-K last-arrival reduction), carved from the sync buffer's spare slots.
  VkDeviceAddress tileCounters(uint32_t count);
  // Optional per-dispatch profiling: timestamps after every dispatch with a label.
  void beginProfile(VkCommandBuffer commands, uint32_t maxDispatches);
  struct ProfileEntry { std::string label; double milliseconds; };
  std::vector<ProfileEntry> endProfile();  // call after the queue is idle
  void setStageLabel(const std::string& label) { stageLabel_ = label; }
  const std::map<std::string, vk::Pipeline>& pipelines() const { return pipelines_; }
  // Replace the SiLU table (65536 f16 codes indexed by f16 input bits).
  void setSiluTable(const std::vector<uint16_t>& table);

 private:
  // requiredSubgroupSize 0: no subgroup requirement (kernels without subgroup operations and a workgroup that is not a
  // multiple of 32)
  VkPipeline pipeline(const char* shader, const vk::SpecConstants& constants, uint32_t requiredSubgroupSize = 32);
  void dispatch(VkCommandBuffer commands, VkPipeline pipeline, const vk::Buffer* const bindings[vk::kGenericBindings],
                const void* push, uint32_t pushBytes, uint32_t x, uint32_t y, uint32_t z);
  void cudaLaunchTracked(VkCommandBuffer commands, VkCudaFunctionNV function, uint32_t gridX, uint32_t gridY,
                         uint32_t gridZ, uint32_t blockX, uint32_t sharedBytes, const void* const* params, size_t paramCount,
                         bool chained = false);
  vk::Buffer syncBuffer_{};
  static constexpr uint32_t kSyncSlotBytes = 6144, kSyncSlots = 96, kTileCounterSlot = 72;   // slots >= 72 hold tile counters
  uint32_t tileCounterCursor_ = 0;
  vk::Buffer chainStatus_{};   // host-visible: [0] waits that timed out, [1] low 32 bits of the first stuck counter's address
  VkDeviceAddress chainStatusAddress();
  struct ChainLaunch { std::string label; VkDeviceAddress waits[2]; VkDeviceAddress signal; };
  std::vector<ChainLaunch> chainLaunches_;   // this recording's PTX launches with counters, in order
  void noteChain(VkDeviceAddress waitA, VkDeviceAddress waitB, VkDeviceAddress signal);
  void dispatchLinear(VkCommandBuffer commands, VkPipeline pipeline,
                      const vk::Buffer* const bindings[vk::kGenericBindings], const void* push, uint32_t pushBytes,
                      uint32_t count);
  vk::Context& context_;
  std::map<std::string, VkShaderModule> modules_;
  std::map<std::string, vk::Pipeline> pipelines_;
  vk::Buffer siluTable_;
  uint32_t dispatches_ = 0;
  VkQueryPool profileQueries_ = VK_NULL_HANDLE;
  uint32_t profileCapacity_ = 0, profileCount_ = 0;
  std::vector<std::string> profileLabels_;
  std::string stageLabel_;
  std::string dispatchLabel_;
};

}  // namespace nr

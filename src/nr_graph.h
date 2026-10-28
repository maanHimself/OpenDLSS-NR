// The NR network graph: 71 blocks over a six-level encoder/decoder with a global
// ViT bottleneck, recorded into one Vulkan command buffer.
#pragma once
#include <functional>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <vector>

#include "kernels.h"
#include "nr_model.h"

namespace nr {

struct Geometry {
  uint32_t validWidth = 0, validHeight = 0;
  uint32_t fullWidth = 0, fullHeight = 0;
  struct Level { uint32_t width, height; } levels[6];
  static Geometry fromValid(uint32_t validWidth, uint32_t validHeight);
  uint32_t vitTokens() const { return levels[5].width * levels[5].height; }
  uint32_t paddedVitTokens() const { return (vitTokens() + 63) & ~63u; }
};

struct FusedLayout {
  uint32_t hidden = 128, heads = 1;
  uint32_t expand = 0, contractWeights = 0, inputAdapter = 0, ffnCosSkip = 0, qkv = 0, relative = 0, scale = 0,
           projection = 0, attnCosSkip = 0, endWithoutPadding = 0;
  uint32_t upsampleWeight = 0, transitionScale = 0, inputScale = 0, adapterScale = 0, postWeights = 0;
  bool expertFfn = false;
  uint32_t expertCount = 0;
};
FusedLayout fusedLayout(uint32_t channels, uint32_t base = 0);
FusedLayout preFusedLayout();
FusedLayout upsampleFusedLayout(uint32_t inputChannels, uint32_t channels);
FusedLayout postFusedLayout();

class Graph {
 public:
  static constexpr uint32_t kBlockCount = 71;   // blocks 0..70; the graph's stage boundaries are hard-coded

  struct Options {
    bool captureBoundaries = false;  // copy every block/transition output for parity checks
    bool captureIntermediates = false;  // also copy the intra-block tensors (FFN, QKV, attention)
    bool fusedBlocks = true;            // one fused dispatch per 32-channel block (false: reference kernels)
  };
  Graph(vk::Context& context, Model& model, Kernels& kernels, const Geometry& geometry, Options options);
  ~Graph();

  // Record the complete network. `inputFeatures` is f32 [fullWidth*fullHeight][16].
  void record(VkCommandBuffer commands, const Activation& inputFeatures);

  const Activation& head() const { return *head_; }  // f32 [full rows][4]
  // The captures (Options::captureBoundaries), by name. Read after a run, outside the graph: like the head, they
  // keep their own buffers when the activations share memory by lifetime.
  const std::map<std::string, Activation*>& boundaries() const { return boundaries_; }
  const Geometry& geometry() const { return geometry_; }
  // Whether consecutive launches are linked by device counters instead of barriers (docs/execution.md).
  bool chained() const { return routes_.chain || routes_.vitChain; }
  // The stored outputs a boundary fixture can hold references for, in graph order: blocks 0-69 and the five encoder
  // stage transitions. Block 70 feeds the head on chip; the pooled-* captures have no reference counterpart.
  static const std::vector<std::string>& referenceBoundaryNames();

  Activation* allocate(const std::string& label, uint32_t rows, uint32_t channels, Format format);

  // Activation memory. Each activation has its own allocation until shareActivations. Record once (a warm-up)
  // between trackLifetimes(true) and trackLifetimes(false): every use of an activation's buffer is noted with the
  // barrier interval it falls in. A barrier orders every launch before it against every one after, chained or not,
  // so two activations whose intervals do not overlap can share bytes. Every use counts, the transfers included (the
  // captures' copies): the recording reports each through the context's use tracker (vk::Context::setUseTracker),
  // so it binds, takes addresses, copies and fills only through the context. An activation whose use went
  // unreported would be taken for unused (and given a 16-byte buffer) or for dead before that use.
  void trackLifetimes(bool on);
  // After the tracked recording has run, with nothing recorded still pending: the activations it used move into one
  // device-local block, largest first, each at the lowest offset clear of every activation alive in an overlapping
  // interval; the ones it never touched (allocated for routes it did not take) get a 16-byte buffer. The head and
  // `keep` (the input features, written outside the graph) keep their own buffers, and so do the captures (read
  // after the run). The block starts zeroed, as the activations did, and is freed with the graph. Returns the bytes
  // of the moved activations before, and of the block.
  std::pair<VkDeviceSize, VkDeviceSize> shareActivations(std::initializer_list<const Activation*> keep = {});

 private:
  // Window phases. Every resolution level runs its own four-phase cycle of half-window shifts; the phase advances
  // once per block at that level in visit order, and a decoder stage continues the count its encoder stage left.
  static constexpr int kFullLevel = 6;   // levels 0..5 are Geometry::levels, 6 is the un-pooled field
  uint32_t windowPhase_[7]{};
  uint32_t takeWindowPhase(int level) { return windowPhase_[level]++; }
  // Which optional fusions and which PTX routes this recording uses. Counter chaining only links PTX launches
  // (the GLSL kernels neither wait nor signal, and a chained launch has no barrier after it), so it is all or
  // nothing: one switch off takes the whole graph back to barriers.
  struct Routes {
    bool fusePre = true, fusePool = true, fuseUpres = true, fusePost = true;
    bool chain = false;         // counter chaining usable at all (the window and expert stages)
    bool vitChain = false;      // counter chaining of the ViT GEMM + attention launches
    int chainMask = 3;          // 1 expert stages, 2 c32 blocks, 4 split-512 GEMMs
    uint32_t deferMax = 128;    // widest expert stage whose projection is fused into the next block's FFN
  };
  static Routes routesFromEnvironment();
  Routes routes_;
  struct Temporaries {
    Activation* ffn = nullptr;           // E4 [rows][hidden or experts*128]
    Activation* ffnNarrow = nullptr;     // E4 [rows][channels] (expert W2 outputs)
    Activation* ffnResidual = nullptr;   // F16 [rows][channels]
    Activation* ffnQuantized = nullptr;  // E4 [rows][channels]
    Activation* ffnQuantized2 = nullptr; // second buffer (alternating blocks) when the projection is deferred into the next FFN
    Activation* qkv = nullptr;           // F16 [rows][3*channels]
    Activation* normalized = nullptr;    // E4 [rows][3*channels]
    Activation* attended = nullptr;      // E4 [rows][channels]
  };
  struct SplitTemporaries {
    Activation* branch = nullptr;   // E4 [rows][512]
    Activation* middle = nullptr;   // E4 [rows][2048]
    Activation* layer0 = nullptr;   // E4 [rows][512]
    Activation* ffnResidual = nullptr;  // E4 [rows][512]
    Activation* qkv = nullptr;      // F16 [rows][1536]
    Activation* normalized = nullptr;  // E4
    Activation* attended = nullptr;    // E4 [rows][512]
  };
  Temporaries createTemporaries(const std::string& label, uint32_t rows, uint32_t channels);
  SplitTemporaries createSplitTemporaries(const std::string& label, uint32_t rows);

  void encodeFusedBlock(VkCommandBuffer commands, Temporaries& temps, const Activation& state, Activation* output,
                        int block, uint32_t channels, uint32_t width, uint32_t height, uint32_t phase,
                        const FusedLayout& layout, const Tensor& tensor, const Activation* ffnSkipOverride,
                        Activation* rawOutput, Activation* pooledOutput = nullptr, uint32_t pooledWidth = 0,
                        bool deferProjection = false, bool firstInStage = true, bool lastInStage = true,
                        uint32_t c32WaitScale = 0, bool c32ChainOut = false);
  // Deferred projection: the previous block's attended tile, FFN publication and projection weights, consumed by
  // the next block's PTX FFN kernel (which computes the block state on chip).
  struct PendingProjection { const Activation* attended = nullptr; const Activation* ffnQ = nullptr; const vk::Buffer* wproj = nullptr; uint32_t auxAttn = 0; const Tensor* tensor = nullptr; Activation* stateOut = nullptr; int block = 0;
                             uint32_t winExpected = 0, shiftY = 0; };
  PendingProjection pendingProjection_{};
  bool hasPendingProjection_ = false;
  // Chained 32-channel blocks: the producer block's window-row counters (consumed by the next fused block).
  struct C32Chain { VkDeviceAddress rows = 0; uint32_t expected = 0, shiftY = 0; bool valid = false; } c32Prev_{};
  // Fill the chain fields of a fused 32-channel block: wait on c32Prev_ (waitScale: 0 same, 1 producer 2x, 2 producer
  // half resolution), signal / chain when a consumer follows; then records this block as the producer.
  void chainBlock32(Kernels::FusedBlock32Args& f, int block, uint32_t waitScale, bool chainOut);
  void encodeSplitBlock(VkCommandBuffer commands, SplitTemporaries& temps, const Activation& state,
                        Activation* output, int block, uint32_t width, uint32_t height, uint32_t phase,
                        Activation* rawOutput, bool firstInStage = true, bool lastInStage = true);
  void encodeVit(VkCommandBuffer commands, Activation& state, uint32_t tokens);
  void capture(VkCommandBuffer commands, const std::string& name, const Activation& source);

  vk::Context& context_;
  Model& model_;
  Kernels& kernels_;
  Geometry geometry_;
  Options options_;
  std::vector<std::unique_ptr<Activation>> activations_;
  std::map<std::string, Activation*> allocationsByKey_;
  std::set<std::string> usedThisRecord_;
  std::map<std::string, Activation*> boundaries_;
  Activation* head_ = nullptr;
  Activation* inputHalf_ = nullptr;
  std::map<VkBuffer, std::pair<uint32_t, uint32_t>> lifetimes_;   // buffer -> first and last barrier interval
  VkDeviceMemory activationBlock_ = VK_NULL_HANDLE;                // shareActivations
  void rebind(Activation& activation, vk::Buffer buffer);         // destroys the buffer it had
};

}  // namespace nr

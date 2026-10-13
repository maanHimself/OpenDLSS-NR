// CUDA implementation of OpenDLSS-NR operations. Derived semantics: upstream MIT.
// browser-compat-v1 uses software F13/F24 reduction and E4M3 publication.
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <thread>
#include "nr_cuda.h"
#include "nr_build_info.h"
#include "kernels/numerics/numerics.cuh"
#include "window_f13_fused.cuh"
#include "gemm_f13.cuh"
#include "attention_f13.cuh"
#include "f16_compat.cuh"
using H = __half;
using namespace nr;
static thread_local std::string error_text;
static void ck(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
#define API_BEGIN try {
#define API_END                                                                                    \
    return 0;                                                                                      \
    }                                                                                              \
    catch (const std::exception &e) {                                                              \
        error_text = e.what();                                                                     \
        return -1;                                                                                 \
    }
struct Context {
    cudaStream_t stream{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    cudaEvent_t start{}, stop{};
    std::vector<void *> allocations;
    size_t allocated = 0;
    int policy = 6;
    bool fast_window = true;
    int device = 0, device_major = 0, device_minor = 0;
    const std::thread::id owner = std::this_thread::get_id();
    H *q{}, *k{}, *v{}, *score{}, *prob{}, *att{}, *recip{};
    bool capturing = false;
    static constexpr int chunk = 256, max_tokens = 1024;
};
// A context stays on the creating host thread and current CUDA device. Reject
// mismatches before touching its stream, buffers, or cleanup state.
static Context *checked_context(void *ctx) {
    if (!ctx)
        throw std::runtime_error("Missing CUDA context");
    auto *c = static_cast<Context *>(ctx);
    if (c->owner != std::this_thread::get_id())
        throw std::runtime_error("CUDA context belongs to another host thread");
    int device = -1;
    ck(cudaGetDevice(&device));
    if (device != c->device)
        throw std::runtime_error("Current CUDA device differs from the context creation device");
    return c;
}
// Teardown must attempt every release even when synchronization or one release
// fails. Store the first error without allocating/throwing during cleanup.
struct CleanupStatus {
    const char *operation = nullptr;
    cudaError_t cuda = cudaSuccess;
    void record(cudaError_t status, const char *op) noexcept {
        if (status != cudaSuccess && !operation) {
            operation = op;
            cuda = status;
        }
    }
};
static CleanupStatus cleanup_context(Context *c) noexcept {
    CleanupStatus result;
    if (!c)
        return result;
    if (c->stream) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        const cudaError_t queried = cudaStreamIsCapturing(c->stream, &capture);
        result.record(queried, "query stream capture");
        if (c->capturing || (queried == cudaSuccess && capture != cudaStreamCaptureStatusNone)) {
            cudaGraph_t abandoned = nullptr;
            const cudaError_t ended = cudaStreamEndCapture(c->stream, &abandoned);
            // An earlier operator error can invalidate capture; ending it still exits
            // capture mode and is expected cleanup, not a second error to mask the first.
            if (ended != cudaErrorStreamCaptureInvalidated)
                result.record(ended, "abort stream capture");
            if (abandoned && abandoned != c->graph)
                result.record(cudaGraphDestroy(abandoned), "destroy abandoned capture");
            c->capturing = false;
        }
        result.record(cudaStreamSynchronize(c->stream), "synchronize stream during cleanup");
    }
    if (c->executable)
        result.record(cudaGraphExecDestroy(c->executable), "destroy executable graph");
    if (c->graph)
        result.record(cudaGraphDestroy(c->graph), "destroy graph");
    for (void *p : c->allocations)
        if (p)
            result.record(cudaFree(p), "free device allocation");
    if (c->start)
        result.record(cudaEventDestroy(c->start), "destroy start event");
    if (c->stop)
        result.record(cudaEventDestroy(c->stop), "destroy stop event");
    if (c->stream)
        result.record(cudaStreamDestroy(c->stream), "destroy stream");
    delete c;
    return result;
}
static void *alloc(Context *c, size_t n) {
    if (n == 0 || n > size_t(3) * 1024 * 1024 * 1024 - c->allocated)
        throw std::runtime_error("3 GiB explicit allocation limit");
    void *p = nullptr;
    ck(cudaMalloc(&p, n));
    try {
        c->allocations.push_back(p);
    } catch (...) {
        cudaFree(p);
        throw;
    }
    c->allocated += n;
    return p;
}
__global__ void convert(const float *a, H *b, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        b[i] = halfRne(a[i]);
}
__global__ void expand(const H *a, float *b, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        b[i] = halfToFloat(a[i]);
}
__global__ void pool_kernel(const H *A, H *O, int iw, int ih, int ow, int oh, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ow * oh * C)
        return;
    int c = i % C, p = i / C, x = (p % ow) * 2, y = (p / ow) * 2;
    H v = halfRne(0.f);
    if (x + 1 < iw && y + 1 < ih) {
        int a = (y * iw + x) * C + c;
        v = mulHalf(addHalf(addHalf(A[a], A[a + C]), addHalf(A[a + iw * C], A[a + iw * C + C])),
                    halfRne(.25f));
    }
    O[i] = publishE4(v);
}
__global__ void up_kernel(const H *A, const H *skip, const H *s, const H *t, H *O, H *raw, int iw,
                          int ow, int oh, int C, int post) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ow * oh * C)
        return;
    int c = i % C, p = i / C, src = (((p / ow) / 2) * iw + (p % ow) / 2) * C + c;
    // Pinned ops.wgsl publishes the FP32 product-plus-add to half. Keeping
    // the FP32 add explicit preserves its midpoint rounding before half/E4.
    H base = post ? mulHalf(A[src], s[c]) : A[src];
    float product = __fmul_rn(halfToFloat(skip[i]), halfToFloat(post ? t[c] : s[c]));
    H v = halfRne(__fadd_rn(halfToFloat(base), product));
    if (raw)
        raw[i] = v;
    O[i] = publishE4(v);
}
__device__ int inverseTile(int p) {
    int t = p >> 4, w = p & 15;
    return ((t >> 1) * 4 + (w >> 2)) * 8 + (t & 1) * 4 + (w & 3);
}
__device__ H norm32(const H *p) {
    // Match pinned WGSL nr_norm_fma: exact half product in FP32, FP32 add,
    // then half publication. True __hfma can differ at a half midpoint.
    H r[16];
    for (int j = 0; j < 16; ++j) {
        float lo = __fmul_rn(halfToFloat(p[j]), halfToFloat(p[j]));
        float hi = halfToFloat(mulHalf(p[j + 16], p[j + 16]));
        float pair = __fadd_rn(lo, hi);
        r[j] = browserHalfRtz(pair);
    }
    for (int d = 8; d; d /= 2)
        for (int j = 0; j < d; ++j)
            r[j] = addHalf(r[j], r[j + d]);
    return browserHalfRtz(__frsqrt_rn(halfToFloat(r[0])));
}
__device__ H norm32_global(const H *p) {
    H r[16];
    for (int j = 0; j < 16; ++j) {
        float lo = __fmul_rn(halfToFloat(p[j]), halfToFloat(p[j]));
        float hi = halfToFloat(mulHalf(p[j + 16], p[j + 16]));
        r[j] = halfRne(__fadd_rn(lo, hi));
    }
    for (int d = 8; d; d /= 2)
        for (int j = 0; j < d; ++j)
            r[j] = addHalf(r[j], r[j + d]);
    return halfRne(rsqrtf(halfToFloat(r[0])));
}
__global__ void window_pack(const H *src, const float *sc, H *Q, H *K, H *V, int width, int height,
                            int heads, int sx, int sy, int first, int count) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count * 64)
        return;
    int b = t / 64, p = t % 64, task = first + b, head = task % heads, window = task / heads,
        wx = (width + sx + 7) / 8;
    int x = (window % wx) * 8 + (p % 8) - sx, y = (window / wx) * 8 + p / 8 - sy;
    H qn = halfRne(0.f);
    const H *base = nullptr;
    if (x >= 0 && y >= 0 && x < width && y < height) {
        base = src + (y * width + x) * heads * 96 + head * 96;
        qn = norm32(base);
    }
    H scale = browserHalfRtz(sc[head]);
    for (int c = 0; c < 32; ++c)
        Q[b * 2048 + p * 32 + c] =
            base ? publishE4(mulHalf(mulHalf(base[c], qn), scale)) : halfRne(0.f);
    int natural = inverseTile(p);
    x = (window % wx) * 8 + (natural % 8) - sx;
    y = (window / wx) * 8 + natural / 8 - sy;
    base = nullptr;
    if (x >= 0 && y >= 0 && x < width && y < height) {
        base = src + (y * width + x) * heads * 96 + head * 96;
        qn = norm32(base + 32);
    }
    for (int c = 0; c < 32; ++c) {
        K[b * 2048 + p * 32 + c] = base ? publishE4(mulHalf(base[32 + c], qn)) : halfRne(0.f);
        V[b * 2048 + p * 32 + c] = base ? publishE4(base[64 + c]) : halfRne(0.f);
    }
}
__device__ H sum64(H *e) {
    H t[8];
    for (int g = 0; g < 8; ++g)
        t[g] = addHalf(addHalf(addHalf(addHalf(e[g], e[g + 8]), addHalf(e[g + 16], e[g + 24])),
                               addHalf(e[g + 32], e[g + 40])),
                       addHalf(e[g + 48], e[g + 56]));
    return addHalf(addHalf(addHalf(addHalf(t[0], t[2]), t[4]), t[6]),
                   addHalf(addHalf(addHalf(t[1], t[3]), t[5]), t[7]));
}
__global__ void softmax_kernel(const H *S, H *P, H *R, int T, int tokens, int count, int global) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count * T)
        return;
    H total = halfRne(0.f);
    H e[64];
    for (int base = 0; base < T; base += 64) {
        for (int k = 0; k < 64; ++k)
            e[k] = global ? vitExpWeight(S[row * T + base + k])
                          : windowExpWeight(S[row * T + base + k]);
        total = addHalf(total, sum64(e));
        for (int k = 0; k < 64; ++k)
            P[row * T + base + k] = e[k];
    }
    if (global && T > tokens)
        total = subHalf(total, halfRne(halfToFloat(vitExpWeight(halfRne(0.f))) * (T - tokens)));
    H rec = !global ? browserHalfRtz(__fdiv_rn(1.f, halfToFloat(total)))
                    : halfRne(1.f / halfToFloat(total));
    R[row] = rec;
    for (int k = 0; k < T; ++k)
        P[row * T + k] = publishE4(global ? P[row * T + k] : mulHalf(P[row * T + k], rec));
}
__global__ void scatter_window(const H *A, H *O, int w, int h, int heads, int sx, int sy, int first,
                               int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count * 2048)
        return;
    int b = i / 2048, p = (i / 32) % 64, c = i % 32, task = first + b, head = task % heads,
        win = task / heads, wx = (w + sx + 7) / 8;
    int x = (win % wx) * 8 + p % 8 - sx, y = (win / wx) * 8 + p / 8 - sy;
    if (x >= 0 && y >= 0 && x < w && y < h)
        O[(y * w + x) * heads * 32 + head * 32 + c] = A[i];
}
__global__ void vit_pack(const H *src, const float *sc, H *Q, H *K, H *V, int tokens, int T,
                         int heads) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= heads * T)
        return;
    int head = i / T, t = i % T;
    H qn = halfRne(0.f), kn = qn;
    const H *b = nullptr;
    if (t < tokens) {
        b = src + t * heads * 96 + head * 96;
        qn = norm32_global(b);
        kn = norm32_global(b + 32);
    }
    for (int c = 0; c < 32; ++c) {
        Q[i * 32 + c] = b ? publishE4(mulHalf(mulHalf(mulHalf(b[c], qn), halfRne(sqrtf(32.f))),
                                              halfRne(sc[head])))
                          : halfRne(0.f);
        K[i * 32 + c] = b ? publishE4(mulHalf(b[32 + c], kn)) : halfRne(0.f);
        V[i * 32 + c] = b ? publishE4(b[64 + c]) : halfRne(0.f);
    }
}
__global__ void scatter_vit(const H *A, H *O, int tokens, int T, int heads) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * heads * 32)
        return;
    int t = i / (heads * 32), h = (i / 32) % heads, c = i % 32;
    O[i] = A[(h * T + t) * 32 + c];
}

extern "C" {
uint32_t nr_abi_version() {
    return NR_CUDA_ABI_VERSION;
}
const char *nr_source_sha256() {
    return NR_SOURCE_SHA256;
}
const char *nr_build_info() {
    return NR_BUILD_INFO;
}
const char *nr_error() {
    return error_text.c_str();
}
int nr_check_context(void *ctx) {
    API_BEGIN
        checked_context(ctx);
    API_END
}
int nr_create(void **result, int policy, int fused) {
    API_BEGIN
        if (!result)
            throw std::runtime_error("Missing context output pointer");
        *result = nullptr;
        if (policy != NR_CUDA_BROWSER_COMPAT_V1)
            throw std::runtime_error("Only browser-compat-v1 is supported by this library");
        if (fused != 0 && fused != 1)
            throw std::runtime_error("Window implementation must be fused (1) or staged (0)");
        int device = 0;
        ck(cudaGetDevice(&device));
        cudaDeviceProp p;
        ck(cudaGetDeviceProperties(&p, device));
        if (p.major != 7 || p.minor != 5)
            throw std::runtime_error("This release requires SM75 on the current CUDA device");
        auto *c = new Context;
        try {
            c->policy = policy;
            c->fast_window = fused != 0;
            c->device = device;
            c->device_major = p.major;
            c->device_minor = p.minor;
            ck(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking));
            ck(cudaEventCreate(&c->start));
            ck(cudaEventCreate(&c->stop));
            const size_t matrix = std::max(Context::chunk * 64 * 32, 32 * Context::max_tokens * 32);
            const size_t scores =
                std::max(Context::chunk * 64 * 64, 32 * Context::max_tokens * Context::max_tokens);
            c->q = (H *)alloc(c, matrix * 2);
            c->k = (H *)alloc(c, matrix * 2);
            c->v = (H *)alloc(c, matrix * 2);
            c->att = (H *)alloc(c, matrix * 2);
            c->score = (H *)alloc(c, scores * 2);
            c->prob = (H *)alloc(c, scores * 2);
            c->recip = (H *)alloc(c, 32 * Context::max_tokens * 2);
            *result = c;
        } catch (...) {
            cleanup_context(c);
            throw;
        }
    API_END
}
// Receipt of the device used to create this context, including when it was not
// ordinal zero. This query does not select a device or allocate GPU memory.
int nr_device_info(void *ctx, int *ordinal, int *major, int *minor) {
    API_BEGIN
        if (!ctx || !ordinal || !major || !minor)
            throw std::runtime_error("Missing device info context or output pointer");
        const auto *c = checked_context(ctx);
        *ordinal = c->device;
        *major = c->device_major;
        *minor = c->device_minor;
    API_END
}
int nr_context_options(void *ctx, int *policy, int *fused) {
    API_BEGIN
        if (!ctx || !policy || !fused)
            throw std::runtime_error("Missing context options context or output pointer");
        const auto *c = checked_context(ctx);
        *policy = c->policy;
        *fused = c->fast_window ? 1 : 0;
    API_END
}
int nr_destroy(void *ctx) {
    API_BEGIN
        const CleanupStatus status = cleanup_context(ctx ? checked_context(ctx) : nullptr);
        if (status.operation)
            throw std::runtime_error(std::string(status.operation) + ": " +
                                     cudaGetErrorString(status.cuda));
    API_END
}
int nr_alloc(void *ctx, void **p, size_t n) {
    API_BEGIN
        auto *c = checked_context(ctx);
        if (!p)
            throw std::runtime_error("Missing allocation output pointer");
        *p = nullptr;
        *p = alloc(c, n);
    API_END
}
int nr_allocated(void *ctx, size_t *bytes) {
    API_BEGIN
        auto *c = checked_context(ctx);
        if (!bytes)
            throw std::runtime_error("Missing allocation count output pointer");
        *bytes = c->allocated;
    API_END
}
int nr_upload(void *ctx, void *dst, const void *src, size_t n) {
    API_BEGIN
        auto *c = checked_context(ctx);
        ck(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, c->stream));
        ck(cudaStreamSynchronize(c->stream));
    API_END
}
int nr_download(void *ctx, void *dst, const void *src, size_t n) {
    API_BEGIN
        auto *c = checked_context(ctx);
        ck(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, c->stream));
        ck(cudaStreamSynchronize(c->stream));
    API_END
}
int nr_copy(void *ctx, void *dst, const void *src, size_t n) {
    API_BEGIN
        ck(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, checked_context(ctx)->stream));
    API_END
}
int nr_sync(void *ctx) {
    API_BEGIN
        ck(cudaStreamSynchronize(checked_context(ctx)->stream));
    API_END
}
int nr_capture_begin(void *ctx) {
    API_BEGIN
        auto *c = checked_context(ctx);
        if (c->graph || c->capturing)
            throw std::runtime_error("Graph already captured or capture active");
        ck(cudaStreamBeginCapture(c->stream, cudaStreamCaptureModeThreadLocal));
        c->capturing = true;
    API_END
}
int nr_capture_end(void *ctx) {
    API_BEGIN
        auto *c = checked_context(ctx);
        if (!c->capturing)
            throw std::runtime_error("No active graph capture");
        cudaGraph_t graph = nullptr;
        cudaError_t ended = cudaStreamEndCapture(c->stream, &graph);
        if (ended == cudaSuccess || ended == cudaErrorStreamCaptureInvalidated)
            c->capturing = false;
        c->graph = graph;
        ck(ended);
        ck(cudaGraphInstantiate(&c->executable, c->graph, 0));
    API_END
}
int nr_timer_start(void *ctx) {
    API_BEGIN
        auto *c = checked_context(ctx);
        ck(cudaEventRecord(c->start, c->stream));
    API_END
}
int nr_timer_stop(void *ctx, float *ms) {
    API_BEGIN
        auto *c = checked_context(ctx);
        ck(cudaEventRecord(c->stop, c->stream));
        ck(cudaEventSynchronize(c->stop));
        ck(cudaEventElapsedTime(ms, c->start, c->stop));
    API_END
}
int nr_replay(void *ctx, float *ms) {
    API_BEGIN
        auto *c = checked_context(ctx);
        if (!c->executable)
            throw std::runtime_error("No graph");
        ck(cudaEventRecord(c->start, c->stream));
        ck(cudaGraphLaunch(c->executable, c->stream));
        ck(cudaEventRecord(c->stop, c->stream));
        ck(cudaEventSynchronize(c->stop));
        ck(cudaEventElapsedTime(ms, c->start, c->stop));
    API_END
}
int nr_convert(void *ctx, const float *a, H *b, size_t n) {
    API_BEGIN
        convert<<<(n + 255) / 256, 256, 0, checked_context(ctx)->stream>>>(a, b, n);
        ck(cudaGetLastError());
    API_END
}
int nr_expand(void *ctx, const H *a, float *b, size_t n) {
    API_BEGIN
        expand<<<(n + 255) / 256, 256, 0, checked_context(ctx)->stream>>>(a, b, n);
        ck(cudaGetLastError());
    API_END
}
int nr_gemm(void *ctx, const H *a, const H *w, H *out, H *raw, const H *res, const H *scale, int M,
            int K, int N, int batches, int broadcast, int silu, int partition, int quant) {
    API_BEGIN
        auto *c = checked_context(ctx);
        if (M % 16 || K % 16 || N % 16 || M <= 0 || K <= 0 || N <= 0 || batches < 1)
            throw std::runtime_error("Unsupported GEMM shape");
        nr::nr_gemm_f13(a, w, out, raw, res, scale, M, K, N, batches, broadcast, silu, partition,
                        quant, c->stream);
    API_END
}
int nr_f16gemm(void *ctx, const H *a, const H *w, H *out, H *quant, int M, int K, int N) {
    API_BEGIN
        if (K % 8 || M <= 0 || K <= 0 || N <= 0)
            throw std::runtime_error("Bad FP16 GEMM shape");
        auto *c = checked_context(ctx);
        nr_f16_compat(a, w, out, quant, M, K, N, c->stream);
        ck(cudaGetLastError());
    API_END
}
int nr_pool(void *ctx, const H *a, H *out, int iw, int ih, int ow, int oh, int C) {
    API_BEGIN
        pool_kernel<<<(ow * oh * C + 255) / 256, 256, 0, checked_context(ctx)->stream>>>(
            a, out, iw, ih, ow, oh, C);
        ck(cudaGetLastError());
    API_END
}
int nr_up(void *ctx, const H *a, const H *skip, const H *s, const H *t, H *out, H *raw, int iw,
          int ow, int oh, int C, int post) {
    API_BEGIN
        up_kernel<<<(ow * oh * C + 255) / 256, 256, 0, checked_context(ctx)->stream>>>(
            a, skip, s, t, out, raw, iw, ow, oh, C, post);
        ck(cudaGetLastError());
    API_END
}
int nr_window(void *ctx, const H *a, const H *prior, const float *sc, H *out, int w, int h,
              int heads, int phase) {
    API_BEGIN
        auto *c = checked_context(ctx);
        if (w <= 0 || h <= 0 || heads <= 0 || heads > 16 || phase < 0 || phase > 3 ||
            int64_t(w) * h > INT32_MAX / (96 * heads))
            throw std::runtime_error("Unsupported window shape");
        int sx = (phase == 1 || phase == 2) ? 4 : 0, sy = (phase == 1 || phase == 3) ? 4 : 0,
            total = ((w + sx + 7) / 8) * ((h + sy + 7) / 8) * heads;
        if (c->fast_window) {
            nr_window_f13_fused(a, prior, sc, out, w, h, heads, phase, c->stream);
            ck(cudaGetLastError());
            return 0;
        }
        for (int first = 0; first < total; first += Context::chunk) {
            int count = std::min(Context::chunk, total - first);
            window_pack<<<(count * 64 + 127) / 128, 128, 0, c->stream>>>(
                a, sc, c->q, c->k, c->v, w, h, heads, sx, sy, first, count);
            nr_qk_f13(c->q, c->k, prior, c->score, 64, heads, first, count, c->stream);
            softmax_kernel<<<(count * 64 + 127) / 128, 128, 0, c->stream>>>(
                c->score, c->prob, c->recip, 64, 64, count, 0);
            nr_pv_f13(c->prob, c->v, c->att, c->recip, 64, count, 0, c->stream);
            scatter_window<<<(count * 2048 + 255) / 256, 256, 0, c->stream>>>(
                c->att, out, w, h, heads, sx, sy, first, count);
        }
        ck(cudaGetLastError());
    API_END
}
int nr_vit(void *ctx, const H *a, const float *sc, H *out, int tokens, int heads) {
    API_BEGIN
        auto *c = checked_context(ctx);
        if (tokens <= 0 || tokens > Context::max_tokens || heads <= 0 || heads > 32)
            throw std::runtime_error("ViT exceeds bounded workspace");
        int T = (tokens + 63) / 64 * 64;
        if (T > Context::max_tokens || heads > 32)
            throw std::runtime_error("ViT exceeds bounded workspace");
        vit_pack<<<(heads * T + 127) / 128, 128, 0, c->stream>>>(a, sc, c->q, c->k, c->v, tokens, T,
                                                                 heads);
        nr_qk_f13(c->q, c->k, nullptr, c->score, T, heads, 0, heads, c->stream);
        softmax_kernel<<<(heads * T + 127) / 128, 128, 0, c->stream>>>(c->score, c->prob, c->recip,
                                                                       T, tokens, heads, 1);
        nr_pv_f13(c->prob, c->v, c->att, c->recip, T, heads, 1, c->stream);
        scatter_vit<<<(tokens * heads * 32 + 255) / 256, 256, 0, c->stream>>>(c->att, out, tokens,
                                                                              T, heads);
        ck(cudaGetLastError());
    API_END
}
}

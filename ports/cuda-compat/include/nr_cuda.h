/* Internal ABI for the matching opendlss_cuda Python package, version 1.
 * This is not a stable third-party embedding API. All tensor pointers are device
 * pointers, except host buffers in upload/download. The caller owns shape and
 * lifetime validation. Use only on the creating host thread/current CUDA device.
 * nr_destroy consumes an owned context even on cleanup failure; an ownership
 * rejection does not consume it. nr_check_context allows a preflight check.
 */
#ifndef NR_CUDA_H
#define NR_CUDA_H
#include <stddef.h>
#include <stdint.h>
#ifdef __CUDACC__
#include <cuda_fp16.h>
typedef __half nr_half;
#else
typedef uint16_t nr_half; /* IEEE binary16 storage; pointers only in this ABI. */
#endif
#define NR_CUDA_ABI_VERSION 1u
#define NR_CUDA_BROWSER_COMPAT_V1 6
#ifdef __cplusplus
extern "C" {
#endif
uint32_t nr_abi_version();
const char *nr_source_sha256();
const char *nr_build_info();
const char *nr_error();
int nr_check_context(void *ctx);
int nr_create(void **result, int policy, int fused);
int nr_device_info(void *ctx, int *ordinal, int *major, int *minor);
int nr_context_options(void *ctx, int *policy, int *fused);
int nr_destroy(void *ctx);
int nr_alloc(void *ctx, void **p, size_t n);
int nr_allocated(void *ctx, size_t *bytes);
int nr_upload(void *ctx, void *dst, const void *src, size_t n);
int nr_download(void *ctx, void *dst, const void *src, size_t n);
int nr_copy(void *ctx, void *dst, const void *src, size_t n);
int nr_sync(void *ctx);
int nr_capture_begin(void *ctx);
int nr_capture_end(void *ctx);
int nr_timer_start(void *ctx);
int nr_timer_stop(void *ctx, float *ms);
int nr_replay(void *ctx, float *ms);
int nr_convert(void *ctx, const float *a, nr_half *b, size_t n);
int nr_expand(void *ctx, const nr_half *a, float *b, size_t n);
int nr_gemm(void *ctx, const nr_half *a, const nr_half *w, nr_half *out, nr_half *raw,
            const nr_half *res, const nr_half *scale, int M, int K, int N, int batches,
            int broadcast, int silu, int partition, int quant);
int nr_f16gemm(void *ctx, const nr_half *a, const nr_half *w, nr_half *out, nr_half *quant, int M,
               int K, int N);
int nr_pool(void *ctx, const nr_half *a, nr_half *out, int iw, int ih, int ow, int oh, int C);
int nr_up(void *ctx, const nr_half *a, const nr_half *skip, const nr_half *s, const nr_half *t,
          nr_half *out, nr_half *raw, int iw, int ow, int oh, int C, int post);
int nr_window(void *ctx, const nr_half *a, const nr_half *prior, const float *sc, nr_half *out,
              int w, int h, int heads, int phase);
int nr_vit(void *ctx, const nr_half *a, const float *sc, nr_half *out, int tokens, int heads);
#ifdef __cplusplus
}
#endif
#endif

// F13 attention with half publication after each K16 group.
#pragma once
#include "f13_dot.cuh"

namespace nr {
namespace attention_f13_detail {
using H = __half;

static __global__ void qk(const H *query, const H *key, const H *prior, H *scores, int tokens,
                          int heads, int first) {
    const int tx = threadIdx.x, ty = threadIdx.y, batch = blockIdx.z;
    const int row = blockIdx.y * 16 + ty, column = blockIdx.x * 16 + tx;
    const int head = (first + batch) % heads;
    H accumulator = prior ? prior[head * 4096 + row * 64 + column] : halfRne(0.f);
    __shared__ __align__(32) H q[16][16], k[16][16];
    for (int channel = 0; channel < 32; channel += 16) {
        q[ty][tx] = query[(size_t(batch) * tokens + row) * 32 + channel + tx];
        // K uses key-major rows so cooperative loads stay contiguous; step16
        // reads one row from K to compute Q*K^T without a global transpose.
        k[ty][tx] = key[(size_t(batch) * tokens + blockIdx.x * 16 + ty) * 32 + channel + tx];
        __syncthreads();
        accumulator = f13_detail::step16(&q[ty][0], 1, &k[tx][0], 1, accumulator);
        __syncthreads();
    }
    scores[(size_t(batch) * tokens + row) * tokens + column] = accumulator;
}

static __global__ void pv(const H *probabilities, const H *values, H *output, const H *reciprocal,
                          int tokens, bool global) {
    const int tx = threadIdx.x, ty = threadIdx.y, batch = blockIdx.z;
    const int row = blockIdx.y * 16 + ty, column = blockIdx.x * 16 + tx;
    H accumulator = halfRne(0.f);
    __shared__ __align__(32) H p[16][16], v[16][16];
    for (int start = 0; start < tokens; start += 16) {
        p[ty][tx] = probabilities[(size_t(batch) * tokens + row) * tokens + start + tx];
        v[ty][tx] = values[(size_t(batch) * tokens + start + ty) * 32 + column];
        __syncthreads();
        accumulator = f13_detail::step16(&p[ty][0], 1, &v[0][tx], 16, accumulator);
        __syncthreads();
    }
    if (global)
        accumulator = mulHalf(accumulator, reciprocal[size_t(batch) * tokens + row]);
    output[(size_t(batch) * tokens + row) * 32 + column] = publishE4(accumulator);
}

inline void shape(int tokens, int count) {
    if (tokens <= 0 || tokens > 1024 || tokens % 16 || count <= 0 || count > 65535)
        throw std::invalid_argument("Invalid staged F13 attention shape");
}
inline void checked_launch() {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace attention_f13_detail

// Q/K are [count,T,32], scores are [count,T,T]. Priors, when present, are
// [heads,64,64] in natural-query/physical-key order. first selects the global
// window/head task offset used by the runtime.
inline void nr_qk_f13(const __half *query, const __half *key, const __half *prior, __half *scores,
                      int T, int heads, int first, int count, cudaStream_t stream) {
    attention_f13_detail::shape(T, count);
    if (!query || !key || !scores || heads <= 0 || heads > 32 || first < 0 ||
        first > INT_MAX - count || (prior && T != 64))
        throw std::invalid_argument("Invalid staged F13 QK arguments");
    attention_f13_detail::qk<<<dim3(T / 16, T / 16, count), dim3(16, 16), 0, stream>>>(
        query, key, prior, scores, T, heads, first);
    attention_f13_detail::checked_launch();
}

// P is [count,T,T], V/output are [count,T,32]. For global attention, reciprocal
// is [count,T] and the final half dot is multiplied by it with half publication
// before E4 publication. Window P already includes normalization, so no scale.
inline void nr_pv_f13(const __half *probabilities, const __half *values, __half *output,
                      const __half *reciprocal, int T, int count, int global, cudaStream_t stream) {
    attention_f13_detail::shape(T, count);
    if (!probabilities || !values || !output || (global && !reciprocal) ||
        (global != 0 && global != 1))
        throw std::invalid_argument("Invalid staged F13 PV arguments");
    attention_f13_detail::pv<<<dim3(2, T / 16, count), dim3(16, 16), 0, stream>>>(
        probabilities, values, output, reciprocal, T, global != 0);
    attention_f13_detail::checked_launch();
}
} // namespace nr

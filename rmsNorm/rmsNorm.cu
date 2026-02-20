// rmsnorm_nocub.cu
// nvcc -O3 -lineinfo rmsnorm_nocub.cu -o rmsnorm_nocub
//
// Non-CUB RMSNorm over dim=1 for x shaped [B, F, D1, D2] (NCHW).
// One block per (b, sdata) where s = d1*D2 + d2.

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(err));                                 \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

template <int BLOCK_THREADS>
__global__ void rmsnorm_dim1_nocub(
    const float* __restrict__ x,
    float* __restrict__ y,
    int B, int F, int D1, int D2,
    float eps
) {
    int b   = (int)blockIdx.y;
    int s   = (int)blockIdx.x;
    int tid = (int)threadIdx.x;

    int S = D1 * D2;
    if (b >= B || s >= S) return;

    int d1 = s / D2;
    int d2 = s % D2;

    // 1) accumulate partial sum of squares for this (b, d1, d2)
    float thread_sum = 0.0f;
    for (int f = tid; f < F; f += BLOCK_THREADS) {
        int idx = ((b * F + f) * D1 + d1) * D2 + d2;
        float v = x[idx];
        thread_sum += v * v;
    }

    // 2) block reduction (shared memory tree reduce)
    __shared__ float smem[BLOCK_THREADS];
    smem[tid] = thread_sum;
    __syncthreads();

    // Reduce smem[0..BLOCK_THREADS-1] to smem[0]
    for (int offset = BLOCK_THREADS / 2; offset > 0; offset >>= 1) {
        if (tid < offset) smem[tid] += smem[tid + offset];
        __syncthreads();
    }

    float sum_sq = smem[0];
    float inv_rms = rsqrtf(sum_sq / (float)F + eps);

    // 3) normalize + store
    for (int f = tid; f < F; f += BLOCK_THREADS) {
        int idx = ((b * F + f) * D1 + d1) * D2 + d2;
        y[idx] = x[idx] * inv_rms;
    }
}

static void cpu_rmsnorm_dim1_ref(
    const std::vector<float>& x,
    std::vector<float>& y,
    int B, int F, int D1, int D2,
    float eps
) {
    int S = D1 * D2;
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            int d1 = s / D2;
            int d2 = s % D2;

            double sum_sq = 0.0;
            for (int f = 0; f < F; ++f) {
                int idx = ((b * F + f) * D1 + d1) * D2 + d2;
                double v = (double)x[idx];
                sum_sq += v * v;
            }
            double inv_rms = 1.0 / std::sqrt(sum_sq / (double)F + (double)eps);

            for (int f = 0; f < F; ++f) {
                int idx = ((b * F + f) * D1 + d1) * D2 + d2;
                y[idx] = (float)(x[idx] * inv_rms);
            }
        }
    }
}

int main() {
    // --- Problem sizes (edit as needed) ---
    const int B  = 2;
    const int F  = 64;
    const int D1 = 512;
    const int D2 = 512;
    const float eps = 1e-5f;

    const int S = D1 * D2;
    const size_t N = (size_t)B * (size_t)F * (size_t)D1 * (size_t)D2;
    const size_t bytes = N * sizeof(float);

    std::printf("Running RMSNorm dim=1 (no CUB): B=%d F=%d D1=%d D2=%d (N=%zu)\n",
                B, F, D1, D2, N);

    // --- Host data init ---
    std::vector<float> h_x(N), h_y(N, 0.0f), h_ref(N, 0.0f);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < N; ++i) h_x[i] = dist(rng);

    // --- Device alloc/copy ---
    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));

    // --- Launch ---
    constexpr int BLOCK = 256; // try 128/256/512 depending on F
    dim3 grid(S, B, 1);
    dim3 block(BLOCK, 1, 1);

    rmsnorm_dim1_nocub<BLOCK><<<grid, block>>>(d_x, d_y, B, F, D1, D2, eps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Copy back ---
    CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

    // --- CPU reference + correctness check ---
    cpu_rmsnorm_dim1_ref(h_x, h_ref, B, F, D1, D2, eps);

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float a = h_ref[i];
        float b = h_y[i];
        float abs_err = std::fabs(a - b);
        float rel_err = abs_err / (std::fabs(a) + 1e-12f);
        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, rel_err);
    }

    std::printf("Max abs err: %.6g\n", max_abs);
    std::printf("Max rel err: %.6g\n", max_rel);

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    return (max_rel < 1e-5f || max_abs < 1e-5f) ? 0 : 2;
}
#include <iostream>
#include <vector>
#include "cuda_utils.hpp"

__global__ void vec_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1 << 20;  // 1M elements
    const size_t bytes = N * sizeof(float);

    std::vector<float> h_a(N), h_b(N), h_c(N);

    for (int i = 0; i < N; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    int block = 256;
    int grid  = (N + block - 1) / block;

    vec_add_kernel<<<grid, block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    std::cout << "c[0] = " << h_c[0] << " (expected 3)\n";
    std::cout << "c[N-1] = " << h_c[N - 1] << " (expected 3)\n";

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return 0;
}

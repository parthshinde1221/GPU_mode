#include <iostream>
#include <vector>
#include "cuda_utils.hpp"

const int TILE = 16;

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        int tiledRow = row;
        int tiledCol = t * TILE + threadIdx.x;
        if (tiledRow < N && tiledCol < N)
            As[threadIdx.y][threadIdx.x] = A[tiledRow * N + tiledCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        tiledRow = t * TILE + threadIdx.y;
        tiledCol = col;
        if (tiledRow < N && tiledCol < N)
            Bs[threadIdx.y][threadIdx.x] = B[tiledRow * N + tiledCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 512;
    const size_t bytes = N * N * sizeof(float);

    std::vector<float> h_A(N * N, 1.0f);
    std::vector<float> h_B(N * N, 2.0f);
    std::vector<float> h_C(N * N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE,
              (N + TILE - 1) / TILE);

    matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));
    std::cout << "[tiled] C[0] = " << h_C[0] << " (expected " << 2.0f * N << ")\n";

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}

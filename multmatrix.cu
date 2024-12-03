#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <ctime>

#define TILE_SIZE 64 // Tile size for chunking

// CUDA Kernel for matrix multiplication of tiles
__global__ void tiledMatrixMulKernel(float *a, float *b, float *c, int n) {
    __shared__ float shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_b[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0f;

    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        if (row < n && (tile * TILE_SIZE + threadIdx.x) < n) {
            shared_a[threadIdx.y][threadIdx.x] = a[row * n + tile * TILE_SIZE + threadIdx.x];
        } else {
            shared_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < n && (tile * TILE_SIZE + threadIdx.y) < n) {
            shared_b[threadIdx.y][threadIdx.x] = b[(tile * TILE_SIZE + threadIdx.y) * n + col];
        } else {
            shared_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            value += shared_a[threadIdx.y][k] * shared_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        c[row * n + col] = value;
    }
}

// Host function for GPU matrix multiplication
void chunkedMatrixMul(float *a, float *b, float *c, int n) {
    float *d_a, *d_b, *d_c;

    size_t size = n * n * sizeof(float);
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, size);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    auto start = std::chrono::high_resolution_clock::now();
    tiledMatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;

    std::cout << std::left << std::setw(15) << (std::to_string(n) + " x " + std::to_string(n))
              << std::setw(15) << duration.count();

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// CPU-based matrix multiplication
void cpuMatrixMul(const float *a, const float *b, float *c, int n) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float value = 0.0f;
            for (int k = 0; k < n; ++k) {
                value += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = value;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

    std::cout << std::setw(15) << duration.count() << std::endl;
}

// Warm-up function to initialize CUDA runtime
void warmupKernel() {
    float *dummy_a, *dummy_b, *dummy_c;
    cudaMalloc((void**)&dummy_a, sizeof(float));
    cudaMalloc((void**)&dummy_b, sizeof(float));
    cudaMalloc((void**)&dummy_c, sizeof(float));
    cudaFree(dummy_a);
    cudaFree(dummy_b);
    cudaFree(dummy_c);
}

int main() {
    const int MAX_SIZE = 32768; // Maximum matrix size
    const int CPU_CUTOFF = 2048; // Cutoff size for CPU computations

    std::cout << std::left << std::setw(15) << "Size"
              << std::setw(15) << "GPU Time (ms)"
              << std::setw(15) << "CPU Time (ms)" << std::endl;
    std::cout << std::string(45, '-') << std::endl;

    warmupKernel();

    for (int size = 2; size <= MAX_SIZE; size *= 2) {
        float *a = (float *)malloc(size * size * sizeof(float));
        float *b = (float *)malloc(size * size * sizeof(float));
        float *gpu_c = (float *)malloc(size * size * sizeof(float));
        float *cpu_c = nullptr;

        if (!a || !b || !gpu_c) {
            std::cerr << "Error allocating host memory for size: " << size << std::endl;
            exit(EXIT_FAILURE);
        }

        if (size <= CPU_CUTOFF) {
            cpu_c = (float *)malloc(size * size * sizeof(float));
            if (!cpu_c) {
                std::cerr << "Error allocating host memory for CPU computation at size: " << size << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        srand(time(NULL));
        for (int i = 0; i < size * size; ++i) {
            a[i] = static_cast<float>(rand()) / RAND_MAX;
            b[i] = static_cast<float>(rand()) / RAND_MAX;
            gpu_c[i] = 0.0f;
            if (cpu_c) cpu_c[i] = 0.0f;
        }

        // GPU calculation
        chunkedMatrixMul(a, b, gpu_c, size);

        // CPU calculation if size <= CPU_CUTOFF
        if (size <= CPU_CUTOFF) {
            cpuMatrixMul(a, b, cpu_c, size);
        } else {
            std::cout << std::setw(15) << "N/A" << std::endl;
        }

        free(a);
        free(b);
        free(gpu_c);
        if (cpu_c) free(cpu_c);
    }

    return 0;
}

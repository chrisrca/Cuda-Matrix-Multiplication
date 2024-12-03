#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <sstream>

#define TILE_SIZE 64 // Tile size for chunking

// CUDA Kernel for matrix multiplication of tiles
__global__ void tiledMatrixMulKernel(float *a, float *b, float *c, int n) {
    __shared__ float shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_b[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0f;

    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tiles into shared memory
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

// Host function for matrix multiplication
void chunkedMatrixMul(float *a, float *b, float *c, int n) {
    float *d_a, *d_b, *d_c;

    // Allocate GPU memory
    size_t size = n * n * sizeof(float);
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy input matrices to GPU
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemset(d_c, 0, size);

    // Launch kernel
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    // Start timing only for the computation
    auto start = std::chrono::high_resolution_clock::now();
    tiledMatrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize(); // Ensure all threads finish before timing ends
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;

    std::cout << std::left << std::setw(15) << (std::to_string(n) + " x " + std::to_string(n))
              << std::setw(15) << duration.count() << std::endl;

    // Copy result matrix back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
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

    std::cout << std::left << std::setw(15) << "Size"
              << std::setw(15) << "Time (ms)" << std::endl;
    std::cout << std::string(25, '-') << std::endl;

    warmupKernel();

    for (int size = 2; size <= MAX_SIZE; size *= 2) {
        // Allocate host memory
        float *a = (float *)malloc(size * size * sizeof(float));
        float *b = (float *)malloc(size * size * sizeof(float));
        float *c = (float *)malloc(size * size * sizeof(float));

        if (!a || !b || !c) {
            std::cerr << "Error allocating host memory for size: " << size << std::endl;
            exit(EXIT_FAILURE);
        }

        // Initialize matrices with random values
        srand(time(NULL));
        for (int i = 0; i < size * size; ++i) {
            a[i] = static_cast<float>(rand()) / RAND_MAX;
            b[i] = static_cast<float>(rand()) / RAND_MAX;
            c[i] = 0.0f;
        }

        // Perform matrix multiplication and time it
        chunkedMatrixMul(a, b, c, size);

        // Free host memory
        free(a);
        free(b);
        free(c);
    }

    return 0;
}

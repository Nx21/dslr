#include "MatrixCuda.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>

// CUDA kernel for matrix addition
__global__ void kernel_matrix_add(double* result, const double* a, const double* b, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

// CUDA kernel for matrix subtraction
__global__ void kernel_matrix_subtract(double* result, const double* a, const double* b, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] - b[idx];
    }
}

// CUDA kernel for matrix multiplication
__global__ void kernel_matrix_multiply(double* result, const double* a, const double* b, 
                                       size_t rows_a, size_t cols_a, size_t cols_b) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows_a && col < cols_b) {
        double sum = 0.0;
        for (size_t k = 0; k < cols_a; ++k) {
            sum += a[row * cols_a + k] * b[k * cols_b + col];
        }
        result[row * cols_b + col] = sum;
    }
}

// CUDA kernel for scalar multiplication
__global__ void kernel_matrix_scalar_multiply(double* result, const double* a, double scalar, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * scalar;
    }
}

// CUDA kernel for scalar division
__global__ void kernel_matrix_scalar_divide(double* result, const double* a, double scalar, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] / scalar;
    }
}

// CUDA kernel for matrix transpose
__global__ void kernel_matrix_transpose(double* result, const double* a, size_t rows, size_t cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        result[col * rows + row] = a[row * cols + col];
    }
}

// CUDA kernel for filling matrix with a value
__global__ void kernel_matrix_fill(double* data, double value, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

// CUDA kernel for randomizing matrix values
__global__ void kernel_matrix_randomize(double* data, double min, double max, size_t size, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = min + (max - min) * curand_uniform_double(&state);
    }
}

// Helper function to calculate grid and block dimensions
void calculateDimensions(size_t size, dim3& blockSize, dim3& gridSize) {
    blockSize = dim3(256);
    gridSize = dim3((size + blockSize.x - 1) / blockSize.x);
}

void calculate2DDimensions(size_t rows, size_t cols, dim3& blockSize, dim3& gridSize) {
    blockSize = dim3(16, 16);
    gridSize = dim3((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);
}

// Wrapper functions for CUDA kernels
extern "C" {
    void cuda_matrix_add(double* result, const double* a, const double* b, size_t size) {
        dim3 blockSize, gridSize;
        calculateDimensions(size, blockSize, gridSize);
        kernel_matrix_add<<<gridSize, blockSize>>>(result, a, b, size);
        cudaDeviceSynchronize();
    }

    void cuda_matrix_subtract(double* result, const double* a, const double* b, size_t size) {
        dim3 blockSize, gridSize;
        calculateDimensions(size, blockSize, gridSize);
        kernel_matrix_subtract<<<gridSize, blockSize>>>(result, a, b, size);
        cudaDeviceSynchronize();
    }

    void cuda_matrix_multiply(double* result, const double* a, const double* b, 
                              size_t rows_a, size_t cols_a, size_t cols_b) {
        dim3 blockSize, gridSize;
        calculate2DDimensions(rows_a, cols_b, blockSize, gridSize);
        kernel_matrix_multiply<<<gridSize, blockSize>>>(result, a, b, rows_a, cols_a, cols_b);
        cudaDeviceSynchronize();
    }

    void cuda_matrix_scalar_multiply(double* result, const double* a, double scalar, size_t size) {
        dim3 blockSize, gridSize;
        calculateDimensions(size, blockSize, gridSize);
        kernel_matrix_scalar_multiply<<<gridSize, blockSize>>>(result, a, scalar, size);
        cudaDeviceSynchronize();
    }

    void cuda_matrix_scalar_divide(double* result, const double* a, double scalar, size_t size) {
        dim3 blockSize, gridSize;
        calculateDimensions(size, blockSize, gridSize);
        kernel_matrix_scalar_divide<<<gridSize, blockSize>>>(result, a, scalar, size);
        cudaDeviceSynchronize();
    }

    void cuda_matrix_transpose(double* result, const double* a, size_t rows, size_t cols) {
        dim3 blockSize, gridSize;
        calculate2DDimensions(rows, cols, blockSize, gridSize);
        kernel_matrix_transpose<<<gridSize, blockSize>>>(result, a, rows, cols);
        cudaDeviceSynchronize();
    }

    void cuda_matrix_fill(double* data, double value, size_t size) {
        dim3 blockSize, gridSize;
        calculateDimensions(size, blockSize, gridSize);
        kernel_matrix_fill<<<gridSize, blockSize>>>(data, value, size);
        cudaDeviceSynchronize();
    }

    void cuda_matrix_randomize(double* data, double min, double max, size_t size, unsigned long long seed) {
        dim3 blockSize, gridSize;
        calculateDimensions(size, blockSize, gridSize);
        kernel_matrix_randomize<<<gridSize, blockSize>>>(data, min, max, size, seed);
        cudaDeviceSynchronize();
    }
}
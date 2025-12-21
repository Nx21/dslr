// CUDA kernels for GPU-accelerated statistics operations
#include <cuda_runtime.h>

__global__ void sumKernel(const double* data, double* result, size_t n) {
    __shared__ double shared_data[256];
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = threadIdx.x;
    
    shared_data[tid] = (idx < n) ? data[idx] : 0.0;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, shared_data[0]);
    }
}

__global__ void varianceKernel(const double* data, double mean_val, double* result, size_t n) {
    __shared__ double shared_data[256];
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tid = threadIdx.x;
    
    if (idx < n) {
        double diff = data[idx] - mean_val;
        shared_data[tid] = diff * diff;
    } else {
        shared_data[tid] = 0.0;
    }
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, shared_data[0]);
    }
}

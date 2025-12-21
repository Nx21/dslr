// GPU-accelerated statistics operations
#include "../include/StatsCuda.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <functional>
#include <iostream>
#include <numeric>

double StatsCuda::meanGPU(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    //todo: Implement GPU mean calculation
}

double StatsCuda::sumGPU(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    //todo: Implement GPU sum calculation
}

double StatsCuda::varianceGPU(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    // todo: Implement GPU variance calculation
}

double StatsCuda::stdDevGPU(const std::vector<double>& data) {
    // todo: Implement GPU standard deviation calculation
}

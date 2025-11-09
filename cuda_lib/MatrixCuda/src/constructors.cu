#include "MatrixCuda.cuh"

void MatrixCuda::checkCudaError(cudaError_t error, const char* operation) const {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in ") + operation + ": " + cudaGetErrorString(error));
    }
}

void MatrixCuda::allocateDevice() {
    if (!on_device && rows > 0 && cols > 0) {
        size_t size = rows * cols * sizeof(double);
        cudaError_t error = cudaMalloc(&d_data, size);
        checkCudaError(error, "cudaMalloc");
        on_device = true;
    }
}

void MatrixCuda::deallocateDevice() {
    if (on_device && d_data != nullptr) {
        cudaFree(d_data);
        d_data = nullptr;
        on_device = false;
    }
}

void MatrixCuda::copyToDevice(const std::vector<std::vector<double>>& host_data) {
    if (rows == 0 || cols == 0) return;
    
    // Flatten the 2D vector to 1D for CUDA memory transfer
    std::vector<double> flattened(rows * cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            flattened[i * cols + j] = host_data[i][j];
        }
    }
    
    allocateDevice();
    size_t size = rows * cols * sizeof(double);
    cudaError_t error = cudaMemcpy(d_data, flattened.data(), size, cudaMemcpyHostToDevice);
    checkCudaError(error, "cudaMemcpy Host to Device");
}

void MatrixCuda::copyFromDevice(std::vector<std::vector<double>>& host_data) const {
    if (!on_device || rows == 0 || cols == 0) return;
    
    // Copy from device to temporary flattened array
    std::vector<double> flattened(rows * cols);
    size_t size = rows * cols * sizeof(double);
    cudaError_t error = cudaMemcpy(flattened.data(), d_data, size, cudaMemcpyDeviceToHost);
    checkCudaError(error, "cudaMemcpy Device to Host");
    
    // Convert flattened array back to 2D vector
    host_data.resize(rows);
    for (size_t i = 0; i < rows; ++i) {
        host_data[i].resize(cols);
        for (size_t j = 0; j < cols; ++j) {
            host_data[i][j] = flattened[i * cols + j];
        }
    }
}

std::vector<std::vector<double>> MatrixCuda::getHostData() const {
    std::vector<std::vector<double>> host_data;
    copyFromDevice(host_data);
    return host_data;
}

MatrixCuda::MatrixCuda() : rows(0), cols(0), d_data(nullptr), on_device(false) {}

MatrixCuda::MatrixCuda(size_t rows, size_t cols, double value) 
    : rows(rows), cols(cols), d_data(nullptr), on_device(false) {
    if (rows > 0 && cols > 0) {
        allocateDevice();
        fill(value);
    }
}

MatrixCuda::MatrixCuda(const std::vector<std::vector<double>>& data) 
    : rows(data.size()), cols(data.empty() ? 0 : data[0].size()), d_data(nullptr), on_device(false) {
    
    // Validate dimensions
    for (const auto& row : data) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
    }
    
    if (rows > 0 && cols > 0) {
        copyToDevice(data);
    }
}

MatrixCuda::MatrixCuda(std::initializer_list<std::initializer_list<double>> init_list) 
    : d_data(nullptr), on_device(false) {
    rows = init_list.size();
    if (rows == 0) {
        cols = 0;
        return;
    }
    
    cols = init_list.begin()->size();
    
    // Convert initializer list to vector
    std::vector<std::vector<double>> host_data;
    host_data.reserve(rows);
    
    for (const auto& row : init_list) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
        host_data.emplace_back(row);
    }
    
    if (rows > 0 && cols > 0) {
        copyToDevice(host_data);
    }
}

MatrixCuda::MatrixCuda(const MatrixCuda& other) 
    : rows(other.rows), cols(other.cols), d_data(nullptr), on_device(false) {
    if (other.on_device && rows > 0 && cols > 0) {
        allocateDevice();
        size_t size = rows * cols * sizeof(double);
        cudaError_t error = cudaMemcpy(d_data, other.d_data, size, cudaMemcpyDeviceToDevice);
        checkCudaError(error, "cudaMemcpy Device to Device (copy constructor)");
    }
}

MatrixCuda& MatrixCuda::operator=(const MatrixCuda& other) {
    if (this != &other) {
        // Clean up current resources
        deallocateDevice();
        
        // Copy dimensions
        rows = other.rows;
        cols = other.cols;
        
        // Copy device data if other has it
        if (other.on_device && rows > 0 && cols > 0) {
            allocateDevice();
            size_t size = rows * cols * sizeof(double);
            cudaError_t error = cudaMemcpy(d_data, other.d_data, size, cudaMemcpyDeviceToDevice);
            checkCudaError(error, "cudaMemcpy Device to Device (assignment)");
        }
    }
    return *this;
}

MatrixCuda::~MatrixCuda() {
    deallocateDevice();
}
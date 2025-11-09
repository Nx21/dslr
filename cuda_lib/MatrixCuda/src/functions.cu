#include "MatrixCuda.cuh"
#include <random>

MatrixCuda MatrixCuda::transpose() const {
    MatrixCuda result(cols, rows);
    
    cuda_matrix_transpose(result.d_data, this->d_data, rows, cols);
    
    return result;
}

MatrixCuda MatrixCuda::dot(const MatrixCuda& other) const {
    return (*this) * other;
}

void MatrixCuda::fill(double value) {
    if (!on_device) {
        allocateDevice();
    }
    
    size_t size = rows * cols;
    cuda_matrix_fill(d_data, value, size);
}

void MatrixCuda::randomize(double min, double max) {
    if (!on_device) {
        allocateDevice();
    }
    
    // Generate a random seed
    static std::random_device rd;
    static std::mt19937 gen(rd());
    unsigned long long seed = gen();
    
    size_t size = rows * cols;
    cuda_matrix_randomize(d_data, min, max, size, seed);
}

void MatrixCuda::print() const {
    std::cout << *this << std::endl;
}

MatrixCuda MatrixCuda::zeros(size_t rows, size_t cols) {
    return MatrixCuda(rows, cols, 0.0);
}

MatrixCuda MatrixCuda::ones(size_t rows, size_t cols) {
    return MatrixCuda(rows, cols, 1.0);
}

MatrixCuda MatrixCuda::identity(size_t size) {
    MatrixCuda result(size, size, 0.0);
    
    // Set diagonal elements to 1.0
    for (size_t i = 0; i < size; ++i) {
        result.set(i, i, 1.0);
    }
    
    return result;
}

void MatrixCuda::syncToDevice() {
    if (!on_device) {
        allocateDevice();
    }
}

void MatrixCuda::syncFromDevice() {
}

MatrixCuda MatrixCuda::copyToHost() const {
    MatrixCuda result(*this);
    return result;
}
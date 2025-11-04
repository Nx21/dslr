#include "cuMatrix/MatrixCuda.h"

double MatrixCuda::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix indices out of range");
    }
    
    if (!on_device) {
        throw std::runtime_error("Matrix data not on device");
    }
    
    // Copy single element from device to host
    double value;
    size_t offset = row * cols + col;
    cudaError_t error = cudaMemcpy(&value, d_data + offset, sizeof(double), cudaMemcpyDeviceToHost);
    checkCudaError(error, "cudaMemcpy single element from device");
    
    return value;
}

void MatrixCuda::set(size_t row, size_t col, double value) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix indices out of range");
    }
    
    if (!on_device) {
        allocateDevice();
    }
    
    // Copy single element from host to device
    size_t offset = row * cols + col;
    cudaError_t error = cudaMemcpy(d_data + offset, &value, sizeof(double), cudaMemcpyHostToDevice);
    checkCudaError(error, "cudaMemcpy single element to device");
}

MatrixCuda MatrixCuda::operator+(const MatrixCuda& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    MatrixCuda result(rows, cols);
    size_t size = rows * cols;
    
    cuda_matrix_add(result.d_data, this->d_data, other.d_data, size);
    
    return result;
}

MatrixCuda MatrixCuda::operator-(const MatrixCuda& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    
    MatrixCuda result(rows, cols);
    size_t size = rows * cols;
    
    cuda_matrix_subtract(result.d_data, this->d_data, other.d_data, size);
    
    return result;
}

MatrixCuda MatrixCuda::operator*(const MatrixCuda& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    MatrixCuda result(rows, other.cols);
    
    cuda_matrix_multiply(result.d_data, this->d_data, other.d_data, rows, cols, other.cols);
    
    return result;
}

MatrixCuda MatrixCuda::operator*(double scalar) const {
    MatrixCuda result(rows, cols);
    size_t size = rows * cols;
    
    cuda_matrix_scalar_multiply(result.d_data, this->d_data, scalar, size);
    
    return result;
}

MatrixCuda MatrixCuda::operator/(double scalar) const {
    if (std::abs(scalar) < 1e-12) {
        throw std::invalid_argument("Division by zero");
    }
    
    MatrixCuda result(rows, cols);
    size_t size = rows * cols;
    
    cuda_matrix_scalar_divide(result.d_data, this->d_data, scalar, size);
    
    return result;
}

MatrixCuda operator*(double scalar, const MatrixCuda& matrix) {
    return matrix * scalar;
}

std::ostream& operator<<(std::ostream& os, const MatrixCuda& matrix) {
    // Get data from device to host for printing
    auto host_data = matrix.getHostData();
    
    for (size_t i = 0; i < matrix.getRows(); ++i) {
        os << "[ ";
        for (size_t j = 0; j < matrix.getCols(); ++j) {
            os << std::setw(8) << std::fixed << std::setprecision(4) << host_data[i][j];
            if (j < matrix.getCols() - 1) os << ", ";
        }
        os << " ]";
        if (i < matrix.getRows() - 1) os << "\n";
    }
    return os;
}
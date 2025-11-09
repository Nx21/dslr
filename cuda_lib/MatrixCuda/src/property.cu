#include "MatrixCuda.cuh"

std::vector<double> MatrixCuda::getRow(size_t row) const {
    if (row >= rows) {
        throw std::out_of_range("Row index out of range");
    }
    
    std::vector<double> result(cols);
    
    // Copy row data from device to host
    size_t offset = row * cols;
    size_t size = cols * sizeof(double);
    cudaError_t error = cudaMemcpy(result.data(), d_data + offset, size, cudaMemcpyDeviceToHost);
    checkCudaError(error, "cudaMemcpy row from device");
    
    return result;
}

std::vector<double> MatrixCuda::getCol(size_t col) const {
    if (col >= cols) {
        throw std::out_of_range("Column index out of range");
    }
    
    std::vector<double> result;
    result.reserve(rows);
    
    // Copy column data element by element (columns are not contiguous in memory)
    for (size_t i = 0; i < rows; ++i) {
        result.push_back((*this)(i, col));
    }
    
    return result;
}

void MatrixCuda::setRow(size_t row, const std::vector<double>& values) {
    if (row >= rows) {
        throw std::out_of_range("Row index out of range");
    }
    if (values.size() != cols) {
        throw std::invalid_argument("Row size must match matrix columns");
    }
    
    if (!on_device) {
        allocateDevice();
    }
    
    // Copy row data from host to device
    size_t offset = row * cols;
    size_t size = cols * sizeof(double);
    cudaError_t error = cudaMemcpy(d_data + offset, values.data(), size, cudaMemcpyHostToDevice);
    checkCudaError(error, "cudaMemcpy row to device");
}

void MatrixCuda::setCol(size_t col, const std::vector<double>& values) {
    if (col >= cols) {
        throw std::out_of_range("Column index out of range");
    }
    if (values.size() != rows) {
        throw std::invalid_argument("Column size must match matrix rows");
    }
    
    if (!on_device) {
        allocateDevice();
    }
    
    // Set column data element by element (columns are not contiguous in memory)
    for (size_t i = 0; i < rows; ++i) {
        set(i, col, values[i]);
    }
}
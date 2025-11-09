#ifndef MATRIX_CUDA_H
#define MATRIX_CUDA_H

#include <vector>
#include <iostream>
#include <initializer_list>
#include <stdexcept>
#include <random>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

class MatrixCuda {
private:
    size_t rows;
    size_t cols;
    double* d_data;  
    bool on_device;  
    
    
    void allocateDevice();
    void deallocateDevice();
    void copyToDevice(const std::vector<std::vector<double>>& host_data);
    void copyFromDevice(std::vector<std::vector<double>>& host_data) const;
    void checkCudaError(cudaError_t error, const char* operation) const;

public:
    // Constructors and destructor
    MatrixCuda();
    MatrixCuda(size_t rows, size_t cols, double value = 0.0);
    MatrixCuda(const std::vector<std::vector<double>>& data);
    MatrixCuda(std::initializer_list<std::initializer_list<double>> init_list);
    MatrixCuda(const MatrixCuda& other);
    MatrixCuda& operator=(const MatrixCuda& other);
    ~MatrixCuda();
    
    // Getters
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    double* getDeviceData() const { return d_data; }
    bool isOnDevice() const { return on_device; }
    
    // Element access (requires copying from device)
    double operator()(size_t row, size_t col) const;
    void set(size_t row, size_t col, double value);
    
    // Operations
    MatrixCuda operator+(const MatrixCuda& other) const;
    MatrixCuda operator-(const MatrixCuda& other) const;
    MatrixCuda operator*(const MatrixCuda& other) const;
    MatrixCuda operator*(double scalar) const;
    MatrixCuda operator/(double scalar) const;
    
    // Functions
    MatrixCuda transpose() const;
    MatrixCuda dot(const MatrixCuda& other) const;
    void fill(double value);
    void randomize(double min = -1.0, double max = 1.0);
    void print() const;
    static MatrixCuda zeros(size_t rows, size_t cols);
    static MatrixCuda ones(size_t rows, size_t cols);
    static MatrixCuda identity(size_t size);
    
    // Vector operations 
    std::vector<double> toVector() const;
    static MatrixCuda fromVector(const std::vector<double>& vec, bool asColumn = true);
    
    // Property operations
    std::vector<double> getRow(size_t row) const;
    std::vector<double> getCol(size_t col) const;
    void setRow(size_t row, const std::vector<double>& values);
    void setCol(size_t col, const std::vector<double>& values);
    
    // CUDA specific methods
    void syncToDevice();
    void syncFromDevice();
    MatrixCuda copyToHost() const;
    std::vector<std::vector<double>> getHostData() const;
};

// Non-member operators
MatrixCuda operator*(double scalar, const MatrixCuda& matrix);
std::ostream& operator<<(std::ostream& os, const MatrixCuda& matrix);

// CUDA kernel declarations
extern "C" {
    void cuda_matrix_add(double* result, const double* a, const double* b, size_t size);
    void cuda_matrix_subtract(double* result, const double* a, const double* b, size_t size);
    void cuda_matrix_multiply(double* result, const double* a, const double* b, 
                              size_t rows_a, size_t cols_a, size_t cols_b);
    void cuda_matrix_scalar_multiply(double* result, const double* a, double scalar, size_t size);
    void cuda_matrix_scalar_divide(double* result, const double* a, double scalar, size_t size);
    void cuda_matrix_transpose(double* result, const double* a, size_t rows, size_t cols);
    void cuda_matrix_fill(double* data, double value, size_t size);
    void cuda_matrix_randomize(double* data, double min, double max, size_t size, unsigned long long seed);
}

#endif
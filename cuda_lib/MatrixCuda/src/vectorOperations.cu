#include "MatrixCuda.cuh"

std::vector<double> MatrixCuda::toVector() const {
    std::vector<double> result;
    
    if (cols == 1) {
        // Column vector
        result.reserve(rows);
        for (size_t i = 0; i < rows; ++i) {
            result.push_back((*this)(i, 0));
        }
    } else if (rows == 1) {
        // Row vector
        result.reserve(cols);
        for (size_t j = 0; j < cols; ++j) {
            result.push_back((*this)(0, j));
        }
    } else {
        throw std::invalid_argument("Matrix must be a vector (single row or column)");
    }
    
    return result;
}

MatrixCuda MatrixCuda::fromVector(const std::vector<double>& vec, bool asColumn) {
    if (asColumn) {
        // Create column vector
        std::vector<std::vector<double>> host_data(vec.size(), std::vector<double>(1));
        for (size_t i = 0; i < vec.size(); ++i) {
            host_data[i][0] = vec[i];
        }
        return MatrixCuda(host_data);
    } else {
        // Create row vector
        std::vector<std::vector<double>> host_data(1, std::vector<double>(vec.size()));
        for (size_t j = 0; j < vec.size(); ++j) {
            host_data[0][j] = vec[j];
        }
        return MatrixCuda(host_data);
    }
}
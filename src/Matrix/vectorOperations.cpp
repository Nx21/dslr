#include "Matrix.h"

std::vector<double> Matrix::toVector() const {
    std::vector<double> result;
    if (cols == 1) {
        result.reserve(rows);
        for (size_t i = 0; i < rows; ++i) {
            result.push_back(data[i][0]);
        }
    } else if (rows == 1) {
        result.reserve(cols);
        for (size_t j = 0; j < cols; ++j) {
            result.push_back(data[0][j]);
        }
    } else {
        throw std::invalid_argument("Matrix must be a vector (single row or column)");
    }
    return result;
}

Matrix Matrix::fromVector(const std::vector<double>& vec, bool asColumn) {
    if (asColumn) {
        Matrix result(vec.size(), 1);
        for (size_t i = 0; i < vec.size(); ++i) {
            result(i, 0) = vec[i];
        }
        return result;
    } else {
        Matrix result(1, vec.size());
        for (size_t j = 0; j < vec.size(); ++j) {
            result(0, j) = vec[j];
        }
        return result;
    }
}
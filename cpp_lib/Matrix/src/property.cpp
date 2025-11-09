#include "Matrix.hpp"

std::vector<double> Matrix::getRow(size_t row) const {
    if (row >= rows) {
        throw std::out_of_range("Row index out of range");
    }
    return data[row];
}

std::vector<double> Matrix::getCol(size_t col) const {
    if (col >= cols) {
        throw std::out_of_range("Column index out of range");
    }
    
    std::vector<double> result;
    result.reserve(rows);
    for (size_t i = 0; i < rows; ++i) {
        result.push_back(data[i][col]);
    }
    return result;
}

void Matrix::setRow(size_t row, const std::vector<double>& values) {
    if (row >= rows) {
        throw std::out_of_range("Row index out of range");
    }
    if (values.size() != cols) {
        throw std::invalid_argument("Row size must match matrix columns");
    }
    data[row] = values;
}

void Matrix::setCol(size_t col, const std::vector<double>& values) {
    if (col >= cols) {
        throw std::out_of_range("Column index out of range");
    }
    if (values.size() != rows) {
        throw std::invalid_argument("Column size must match matrix rows");
    }
    
    for (size_t i = 0; i < rows; ++i) {
        data[i][col] = values[i];
    }
}

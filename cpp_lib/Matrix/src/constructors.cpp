#include "Matrix.hpp"

Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(size_t rows, size_t cols, double value) 
    : rows(rows), cols(cols), data(rows, std::vector<double>(cols, value)) {}

Matrix::Matrix(const std::vector<std::vector<double>>& data) 
    : rows(data.size()), cols(data.empty() ? 0 : data[0].size()), data(data)  {
    for (const auto& row : data) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
    }
}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> init_list) {
    rows = init_list.size();
    if (rows == 0) {
        cols = 0;
        return;
    }
    
    cols = init_list.begin()->size();
    data.reserve(rows);
    
    for (const auto& row : init_list) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
        data.emplace_back(row);
    }
}
 
Matrix::Matrix(const Matrix& other) 
    : rows(other.rows), cols(other.cols), data(other.data) {}
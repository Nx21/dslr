#include "Matrix.hpp"

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = data[i][j];
        }
    }
    return result;
}

Matrix Matrix::dot(const Matrix& other) const {
    return (*this) * other;
}

void Matrix::fill(double value) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = value;
        }
    }
}

void Matrix::randomize(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = dis(gen);
        }
    }
}

void Matrix::print() const {
    std::cout << *this << std::endl;
}

Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::ones(size_t rows, size_t cols) {
    return Matrix(rows, cols, 1.0);
}

Matrix Matrix::identity(size_t size) {
    Matrix result(size, size, 0.0);
    for (size_t i = 0; i < size; ++i) {
        result(i, i) = 1.0;
    }
    return result;
}

Matrix Matrix::inverse() const {
    if (rows != cols) {
        throw std::invalid_argument("Only square matrices can be inverted.");
    }
    
    size_t n = rows;
    Matrix augmented(n, 2 * n);
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            augmented(i, j) = data[i][j];
            augmented(i, j + n) = (i == j) ? 1.0 : 0.0;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        double diagElement = augmented(i, i);
        if (diagElement == 0) {
            throw std::runtime_error("Matrix is singular and cannot be inverted.");
        }
        for (size_t j = 0; j < 2 * n; ++j) {
            augmented(i, j) /= diagElement;
        }

        for (size_t k = i + 1; k < n; ++k) {
            double factor = augmented(k, i);
            for (size_t j = 0; j < 2 * n; ++j) {
                augmented(k, j) -= factor * augmented(i, j);
            }
        }
    }

    for (int i = n - 1; i >= 0; --i) {
        for (int k = i - 1; k >= 0; --k) {
            double factor = augmented(k, i);
            for (size_t j = 0; j < 2 * n; ++j) {
                augmented(k, j) -= factor * augmented(i, j);
            }
        }
    }
    
    // Extract the right half as the inverse
    Matrix inverse(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            inverse(i, j) = augmented(i, j + n);
        }
    }
    return inverse;
}
#include "Vector.hpp"
#include <stdexcept>

Vector::Vector(size_t size, double value)
    : Matrix(size, 1, value) {}

Vector::Vector(const std::vector<double>& data)
    : Matrix(Matrix::fromVector(data)) {}

Vector::Vector(std::initializer_list<double> init_list)
    : Matrix(Matrix::fromVector(std::vector<double>(init_list))) {}

double& Vector::operator()(size_t index) {
    return Matrix::operator()(index, 0);
}

const double& Vector::operator()(size_t index) const {
    return Matrix::operator()(index, 0);
}

size_t Vector::size() const {
    return getRows();
}

Vector Vector::operator+(const Matrix& other) const {
    if (size() != other.getRows() || other.getCols() != 1) {
        throw std::invalid_argument("Vector sizes do not match for addition");
    }
    Vector result(size());
    for (size_t i = 0; i < size(); ++i) {
        result(i) = (*this)(i) + other(i, 0);
    }
    return result;
}

Vector Vector::operator-(const Matrix& other) const {
    if (size() != other.getRows() || other.getCols() != 1) {
        throw std::invalid_argument("Vector sizes do not match for subtraction");
    }
    Vector result(size());
    for (size_t i = 0; i < size(); ++i) {
        result(i) = (*this)(i) - other(i, 0);
    }
    return result;
}

Vector Vector::operator*(Matrix other) const {
    if (size() != other.getRows()) {
        throw std::invalid_argument("Vector and matrix sizes do not match for multiplication");
    }
    Vector result(other.getCols());
    for (size_t j = 0; j < other.getCols(); ++j) {
        double sum = 0.0;
        for (size_t i = 0; i < size(); ++i) {
            sum += (*this)(i) * other(i, j);
        }
        result(j) = sum;
    }
    return result;
}

Vector operator*(const Matrix& matrix, const Vector& other) {
    return matrix * other;
}

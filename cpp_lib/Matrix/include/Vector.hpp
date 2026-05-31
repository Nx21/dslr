#ifndef VECTOR_H
#define VECTOR_H

#include "Matrix.hpp"
#include <vector>
#include <stdexcept>

class Vector : public Matrix {
public:
    Vector(size_t size, double value = 0.0);
    Vector(const std::vector<double>& data);
    Vector(std::initializer_list<double> init_list);
    
    double& operator()(size_t index);
    const double& operator()(size_t index) const;
    size_t size() const;

    Vector operator+(const Matrix& other) const;
    Vector operator-(const Matrix& other) const;
    Vector operator*(Matrix other) const;
};
Vector operator*(const Matrix& matrix, const Vector& other);
#endif

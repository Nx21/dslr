#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <initializer_list>
#include <stdexcept>
#include <iostream>
#include <random>
#include <iomanip>
#include <cmath>
#include <algorithm>

class Matrix {
private:
    size_t rows;
    size_t cols;
    std::vector<std::vector<double>> data;

public:
    Matrix();
    Matrix(size_t rows, size_t cols, double value = 0.0);
    Matrix(const std::vector<std::vector<double>>& data);
    Matrix(std::initializer_list<std::initializer_list<double>> init_list);
    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);
    ~Matrix(){};
    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;
    
    //  operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;
    
    //  functions
    Matrix transpose() const;
    Matrix dot(const Matrix& other) const;
    void fill(double value);
    void randomize(double min = -1.0, double max = 1.0);
    void print() const;
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix ones(size_t rows, size_t cols);
    static Matrix identity(size_t size);
    
    // Vector operations 
    std::vector<double> toVector() const;
    static Matrix fromVector(const std::vector<double>& vec, bool asColumn = true);
    
    // property operations
    std::vector<double> getRow(size_t row) const;
    std::vector<double> getCol(size_t col) const;
    void setRow(size_t row, const std::vector<double>& values);
    void setCol(size_t col, const std::vector<double>& values);
};

// Non-member operators
Matrix operator*(double scalar, const Matrix& matrix);
std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

#endif
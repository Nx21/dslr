#include "LinearRegression.hpp"

Matrix LinearRegression::addBiasColumn(const Matrix& X) {
    size_t m = X.getRows();
    size_t n = X.getCols();
    
    Matrix X_b(m, n + 1);
    
    for (size_t i = 0; i < m; ++i) {
        X_b(i, 0) = 1.0; // Bias term
        for (size_t j = 0; j < n; ++j) {
            X_b(i, j + 1) = X(i, j);
        }
    }
    
    return X_b;
}

double LinearRegression::sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}


std::vector<double> LinearRegression::computeGradient(const Matrix& X, const std::vector<double>& y, const std::vector<double>& predictions) {
    size_t m = X.getRows();
    size_t n = X.getCols();
    
    std::vector<double> gradients(n, 0.0);
    
    for (size_t j = 0; j < n; ++j) {
        double gradientSum = 0.0;
        for (size_t i = 0; i < m; ++i) {
            gradientSum += (predictions[i] - y[i]) * X(i, j);
        }
        gradients[j] = gradientSum / m;
    }
    
    return gradients;
}
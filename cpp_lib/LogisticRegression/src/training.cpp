#include "LinearRegression.hpp"

void LinearRegression::trainNormalEquation(const Matrix& X, const Vector& y) {
    Matrix X_b = addBiasColumn(X);

    Matrix X_transpose = X_b.transpose();
    Matrix X_transpose_X = X_transpose * X_b;
    Vector X_transpose_y = X_transpose * Vector(y);

    _coefficients = X_transpose_X.inverse() * X_transpose_y;
    _bias = _coefficients(0);   
}

void LinearRegression::trainGradientDescent(const Matrix& X, const Vector& y) {
    Matrix X_b = addBiasColumn(X);
    size_t m = X_b.getRows();
    size_t n = X_b.getCols();

    _coefficients = Vector(n, 0.0);
    _trainingLoss.clear();

    for (size_t iter = 0; iter < _maxIterations; ++iter) {
        Vector predictions = predict(X);
        Vector gradients = computeGradient(X_b, y, predictions);
        for (size_t j = 0; j < n; ++j) {
            _coefficients(j) -= _learningRate * gradients(j);
        }

        double cost = computeCost(X_b, y, predictions);
        _trainingLoss.push_back(cost);
    }
}

void LinearRegression::trainSGD(const Matrix& X, const Vector& y) {
    Matrix X_b = addBiasColumn(X);
    size_t m = X_b.getRows();
    size_t n = X_b.getCols();

    _coefficients = Vector(n, 0.0);
    _trainingLoss.clear();

    for (size_t iter = 0; iter < _maxIterations; ++iter) {
        for (size_t i = 0; i < m; ++i) {
            Vector prediction(1);
            for (size_t j = 0; j < n; ++j) {
                prediction(0) += _coefficients(j) * X_b(i, j);
            }
            double error = prediction(0) - y(i);
            for (size_t j = 0; j < n; ++j) {
                _coefficients(j) -= _learningRate * error * X_b(i, j);
            }
        }

        Vector predictions = predict(X);
        double cost = computeCost(X_b, y, predictions);
        _trainingLoss.push_back(cost);
    }
}



#include "LinearRegression.hpp"

void LinearRegression::trainNormalEquation(const Matrix& X, const std::vector<double>& y) {
    Matrix X_b = addBiasColumn(X);

    Matrix X_transpose = X_b.transpose();
    Matrix X_transpose_X = X_transpose.dot(X_b);
    Matrix X_transpose_y = X_transpose.dot(Matrix::fromVector(y, true));

    _coefficients = X_transpose_X.inverse().dot(X_transpose_y);
    _bias = _coefficients(0, 0);   
}

void LinearRegression::trainGradientDescent(const Matrix& X, const std::vector<double>& y) {
    Matrix X_b = addBiasColumn(X);
    size_t m = X_b.getRows();
    size_t n = X_b.getCols();

    _coefficients = Matrix::zeros(n, 1);
    _trainingLoss.clear();

    for (size_t iter = 0; iter < _maxIterations; ++iter) {
        std::vector<double> predictions = predict(X);
        std::vector<double> gradients = computeGradient(X_b, y, predictions);

        for (size_t j = 0; j < n; ++j) {
            _coefficients(j, 0) -= _learningRate * gradients[j];
        }

        double cost = computeCost(X_b, y, predictions);
        _trainingLoss.push_back(cost);
    }
}



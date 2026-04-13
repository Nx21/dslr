/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   LinearRegression.hpp                               :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: nhanafi <nhanafi@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/04/13 22:43:16 by nhanafi           #+#    #+#             */
/*   Updated: 2026/04/13 23:50:32 by nhanafi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */


#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include "Matrix/include/Matrix.hpp"
#include "Stats/include/Stats.hpp"
#include <vector>
#include <string>

class LinearRegression {
private:
    Matrix _coefficients;           // Learned coefficients (weights and bias)
    double _bias;                   // Intercept term
    bool _isTrainedBiasSeparately;  // Flag to track bias handling
    std::vector<double> _trainingLoss;  // Loss history for monitoring
    
    // Hyperparameters
    double _learningRate;
    size_t _maxIterations;
    double _regularizationParam;    // Lambda for L2 regularization
    bool _useRegularization;

public:
    // Constructor and Destructor
    LinearRegression();
    LinearRegression(double learningRate, size_t maxIterations = 1000);
    LinearRegression(double learningRate, size_t maxIterations, 
                     double regularizationParam, bool useRegularization);
    ~LinearRegression();

    // Training methods
    // Normal Equation: θ = (X^T * X)^(-1) * X^T * y
    void trainNormalEquation(const Matrix& X, const std::vector<double>& y);
    
    // Gradient Descent: θ = θ - α * ∇J(θ)
    void trainGradientDescent(const Matrix& X, const std::vector<double>& y);
    
    // Stochastic Gradient Descent: Updates one sample at a time
    void trainSGD(const Matrix& X, const std::vector<double>& y);
    
    // Mini-batch Gradient Descent
    void trainMiniBatchGD(const Matrix& X, const std::vector<double>& y, size_t batchSize);

    // Prediction methods
    std::vector<double> predict(const Matrix& X) const;
    double predictSingle(const std::vector<double>& x) const;

    // Evaluation metrics
    double mse(const Matrix& X, const std::vector<double>& y) const;
    double rmse(const Matrix& X, const std::vector<double>& y) const;
    double mae(const Matrix& X, const std::vector<double>& y) const;
    double r2Score(const Matrix& X, const std::vector<double>& y) const;

    // Getters
    Matrix getCoefficients() const { return coefficients; }
    double getBias() const { return bias; }
    std::vector<double> getTrainingLoss() const { return trainingLoss; }
    double getLearningRate() const { return learningRate; }
    size_t getMaxIterations() const { return maxIterations; }

    // Setters
    void setLearningRate(double lr) { learningRate = lr; }
    void setMaxIterations(size_t iterations) { maxIterations = iterations; }
    void setRegularization(double lambda, bool use) {
        regularizationParam = lambda;
        useRegularization = use;
    }

    // Utility
    void printCoefficients() const;
    void printMetrics(const Matrix& X, const std::vector<double>& y) const;

private:
    // Helper methods
    static Matrix addBiasColumn(const Matrix& X);
    static double computeCost(const Matrix& X, const std::vector<double>& y, 
                      const std::vector<double>& predictions) ;
    static std::vector<double> computeGradient(const Matrix& X, const std::vector<double>& y,
                                        const std::vector<double>& predictions) ;
    static double sigmoid(double z);
};

#endif

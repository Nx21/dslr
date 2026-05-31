/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   LinearRegression.hpp                               :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: nhanafi <nhanafi@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2026/04/13 22:43:16 by nhanafi           #+#    #+#             */
/*   Updated: 2026/05/31 16:44:17 by nhanafi          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */


#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include "Matrix/include/Vector.hpp"
#include "Stats/include/Stats.hpp"
#include <vector>
#include <string>

class LinearRegression {
private:
    Vector _coefficients;
    double _bias;
    bool _isTrainedBiasSeparately;
    std::vector<double> _trainingLoss;

    // Hyperparameters
    double _learningRate;
    size_t _maxIterations;
    double _regularizationParam;
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
    void trainNormalEquation(const Matrix& X, const Vector& y);
    
    // Gradient Descent: θ = θ - α * ∇J(θ)
    void trainGradientDescent(const Matrix& X, const Vector& y);
    
    // Stochastic Gradient Descent: Updates one sample at a time
    void trainSGD(const Matrix& X, const Vector& y);
    
    // Mini-batch Gradient Descent
    void trainMiniBatchGD(const Matrix& X, const Vector& y, size_t batchSize);

    // Prediction methods
    Vector predict(const Matrix& X) const;
    double predictSingle(const Vector& x) const;

    // Evaluation metrics
    double mse(const Matrix& X, const Vector& y) const;
    double rmse(const Matrix& X, const Vector& y) const;
    double mae(const Matrix& X, const Vector& y) const;
    double r2Score(const Matrix& X, const Vector& y) const;

    // Getters
    Vector getCoefficients() const { return _coefficients; }
    double getBias() const { return _bias; }
    Vector getTrainingLoss() const { return _trainingLoss; }
    double getLearningRate() const { return _learningRate; }
    size_t getMaxIterations() const { return _maxIterations; }

    // Setters
    void setLearningRate(double lr) { _learningRate = lr; }
    void setMaxIterations(size_t iterations) { _maxIterations = iterations; }
    void setRegularization(double lambda, bool use) {
        _regularizationParam = lambda;
        _useRegularization = use;
    }

    // Utility
    void printCoefficients() const;
    void printMetrics(const Matrix& X, const Vector& y) const;

private:
    // Helper methods
    static Matrix addBiasColumn(const Matrix& X);
    static double computeCost(const Matrix& X, const Vector& y, 
                      const Vector& predictions) ;
    static Vector computeGradient(const Matrix& X, const Vector& y,
                                        const Vector& predictions) ;
    static double sigmoid(double z);
    static Vector predict(const Matrix& X, const Vector& coefficients, double bias);

};

#endif

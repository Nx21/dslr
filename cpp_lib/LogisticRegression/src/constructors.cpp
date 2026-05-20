#include "include/LinearRegression.hpp"
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>

// Constructor
LinearRegression::LinearRegression()
    : _bias(0.0), _isTrainedBiasSeparately(false), _learningRate(0.01),
      _maxIterations(1000), _regularizationParam(0.0), _useRegularization(false) {}

LinearRegression::LinearRegression(double lr, size_t maxIter)
    : _bias(0.0), _isTrainedBiasSeparately(false), _learningRate(lr),
      _maxIterations(maxIter), _regularizationParam(0.0), _useRegularization(false) {}

LinearRegression::LinearRegression(double lr, size_t maxIter,
                                   double regParam, bool useReg)
    : _bias(0.0), _isTrainedBiasSeparately(false), _learningRate(lr),
      _maxIterations(maxIter), _regularizationParam(regParam),
      _useRegularization(useReg) {}

LinearRegression::~LinearRegression() {}


# Linear Regression

### Phase 0: Mathematical Foundation  ⏳ IN PROGRESS
- [ ] Problem formulation and cost function definition
- [ ] Normal Equation derivation and implementation
- [ ] Gradient Descent algorithm and theory
- [ ] Stochastic Gradient Descent (SGD) algorithm
- [ ] Mini-Batch Gradient Descent implementation
- [ ] L2 Regularization (Ridge Regression)

### Phase 1: Core Implementation 📋 PENDING
- [ ] Constructor with configurable hyperparameters
- [ ] Training methods: Normal Equation, Batch GD, SGD, Mini-Batch GD
- [ ] Prediction methods: batch and single sample
- [ ] Coefficient management and initialization
- [ ] Convergence tracking and iteration monitoring
- [ ] L2 Regularization parameter support

### Phase 2: Evaluation & Metrics 📋 PENDING
- [ ] Mean Squared Error (MSE) calculation
- [ ] Root Mean Squared Error (RMSE)
- [ ] Mean Absolute Error (MAE)
- [ ] R² Score (Coefficient of Determination)
- [ ] Metric reporting and analysis
- [ ] Training loss history tracking

### Phase 3: Documentation & Examples 📋 PENDING
- [ ] Comprehensive mathematical documentation
- [ ] Algorithm explanations with formulas
- [ ] API reference documentation
- [ ] Build system documentation
- [ ] Example usage code
- [ ] Implementation notes and complexity analysis

### Phase 4: Enhanced Testing  📋 PENDING
- [ ] Unit tests for each training method
- [ ] Cross-validation implementation
- [ ] Performance benchmarks comparing all 4 algorithms
- [ ] Numerical accuracy tests against reference implementations

### Phase 5: Advanced Regularization 📋 PENDING
- [ ] L1 Regularization (Lasso) implementation
- [ ] Elastic Net (L1 + L2) support
- [ ] Automatic regularization parameter tuning
- [ ] Early stopping mechanism

### Phase 6: Optimization & GPU Support  📋 PENDING
- [ ] GPU acceleration for large-scale training
- [ ] CUDA kernels for matrix operations
- [ ] Memory-efficient batch processing
- [ ] Performance profiling and optimization

### Phase 7: Feature Expansion 📋 PENDING
- [ ] Polynomial feature generation
- [ ] Feature selection/elimination
- [ ] Data preprocessing utilities
- [ ] Model persistence (save/load)

A comprehensive C++ implementation of **Linear Regression** using the Matrix and Stats libraries. This module provides multiple training algorithms including the Normal Equation, Gradient Descent, Stochastic Gradient Descent (SGD), and Mini-Batch Gradient Descent with optional L2 regularization.

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Foundation](#mathematical-foundation)
   - [Problem Formulation](#problem-formulation)
   - [Normal Equation](#normal-equation)
   - [Gradient Descent](#gradient-descent)
   - [Stochastic Gradient Descent](#stochastic-gradient-descent)
   - [Mini-Batch Gradient Descent](#mini-batch-gradient-descent)
   - [Regularization (Ridge Regression)](#regularization-ridge-regression)
3. [Evaluation Metrics](#evaluation-metrics)
4. [API Reference](#api-reference)
5. [Usage Examples](#usage-examples)

---

## Overview

Linear Regression is a fundamental supervised learning algorithm that models the linear relationship between input features and continuous output values. This implementation provides four training methods optimized for different scenarios:

- **Normal Equation**: Closed-form solution, ideal for small to medium datasets
- **Gradient Descent**: Batch optimization, suitable for large datasets
- **Stochastic Gradient Descent**: Online learning, memory-efficient
- **Mini-Batch Gradient Descent**: Balance between batch and stochastic approaches

---

## Mathematical Foundation

### Problem Formulation

Given a dataset with $m$ samples, $n$ features, and target values:
- Input matrix: $X \in \mathbb{R}^{m \times n}$
- Target vector: $y \in \mathbb{R}^{m}$
- Coefficient vector: $\theta \in \mathbb{R}^{n+1}$ (includes bias term $\theta_0$)

The linear regression hypothesis function is:

$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n = \theta^T \mathbf{x}$$

Where $\mathbf{x} = [1, x_1, x_2, \ldots, x_n]^T$ (augmented with bias term).

#### Cost Function (Mean Squared Error)

The objective is to minimize the Mean Squared Error (MSE) cost function:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 = \frac{1}{2m} \|X\theta - y\|^2$$

This measures the average squared difference between predicted and actual values.

---

### Normal Equation

The Normal Equation provides a **closed-form solution** by solving the linear system directly.

#### Derivation

Taking the gradient of the cost function and setting it to zero:

$$\nabla J(\theta) = \frac{1}{m} X^T(X\theta - y) = 0$$

Solving for $\theta$:

$$X^T X \theta = X^T y$$

$$\theta = (X^T X)^{-1} X^T y$$

#### Characteristics

- **Computational Complexity**: $O(n^3)$ due to matrix inversion
- **Advantages**:
  - No hyperparameter tuning (no learning rate)
  - No iterations needed
  - Closed-form solution guarantees optimality
- **Disadvantages**:
  - Computationally expensive for large $n$ (features)
  - Matrix inversion may be ill-conditioned
  - Memory intensive for high-dimensional data

#### Implementation Note

In our implementation, we use an iterative approach to approximate the normal equation solution through gradient descent for numerical stability.

---

### Gradient Descent

Gradient Descent is an **iterative optimization algorithm** that incrementally updates parameters towards the minimum of the cost function.

#### Algorithm

The update rule for each iteration is:

$$\theta := \theta - \alpha \nabla J(\theta)$$

Where:
- $\alpha$ is the learning rate (step size)
- $\nabla J(\theta)$ is the gradient of the cost function

#### Gradient Computation

The gradient for each parameter $\theta_j$ is:

$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

In matrix form:

$$\nabla J(\theta) = \frac{1}{m} X^T (X\theta - y)$$

#### Update Step

$$\theta_j := \theta_j - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

#### Characteristics

- **Computational Complexity**: $O(mn)$ per iteration
- **Advantages**:
  - Scalable to large datasets
  - Efficient memory usage
  - Flexible hyperparameter control
- **Disadvantages**:
  - Requires careful learning rate tuning
  - Convergence depends on learning rate and cost function landscape
  - May converge slowly for ill-conditioned problems

#### Convergence Criteria

The algorithm stops when:
1. Maximum iterations reached, or
2. Cost change is below threshold: $|J(\theta)^{(t)} - J(\theta)^{(t-1)}| < \epsilon$

---

### Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) updates parameters using **one sample at a time** instead of the entire batch.

#### Algorithm

For each iteration:

$$\text{For } i = 1 \text{ to } m:$$
$$\theta := \theta - \alpha \cdot (h_\theta(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$$

#### Comparison with Batch Gradient Descent

| Aspect | Batch GD | SGD |
|--------|----------|-----|
| **Update Frequency** | Once per epoch | Once per sample |
| **Noise** | Lower (stable) | Higher (noisy) |
| **Convergence** | Smooth | Fluctuating |
| **Memory** | $O(mn)$ | $O(n)$ |
| **Speed** | Slower iterations | Faster iterations |

#### Characteristics

- **Computational Complexity**: $O(n)$ per sample
- **Advantages**:
  - Very memory efficient
  - Fast initial convergence
  - Can escape local minima due to noise
  - Suitable for streaming data
- **Disadvantages**:
  - Noisier convergence path
  - Harder to parallelize
  - May never fully converge

---

### Mini-Batch Gradient Descent

Mini-Batch GD combines benefits of batch and stochastic approaches by using **small batches** of samples.

#### Algorithm

For each epoch, partition data into batches of size $B$:

$$\text{For each batch } [x^{(i)}, \ldots, x^{(i+B-1)}]:$$
$$\theta := \theta - \alpha \cdot \frac{1}{B} \sum_{j=0}^{B-1} (h_\theta(x^{(i+j)}) - y^{(i+j)}) \cdot x^{(i+j)}$$

#### Characteristics

- **Computational Complexity**: $O(Bn)$ per batch update
- **Batch Size Tradeoffs**:
  - Small batch (e.g., 16-32): More noise, escapes local minima better
  - Large batch (e.g., 256-512): Stable updates, better hardware utilization
  - Typical: 32-128 samples

- **Advantages**:
  - Balance between stability and efficiency
  - Parallelizable
  - Better hardware utilization than SGD
  - Can use vectorized operations

---

### Regularization (Ridge Regression)

Regularization prevents overfitting by penalizing large coefficient values.

#### L2 Regularization (Ridge Regression)

Modified cost function:

$$J(\theta) = \frac{1}{2m} \left( \|X\theta - y\|^2 + \lambda \|\theta_{1:n}\|^2 \right)$$

Where:
- $\lambda$ is the regularization parameter
- $\|\theta_{1:n}\|^2$ excludes the bias term: $\sum_{j=1}^{n} \theta_j^2$

#### Gradient with Regularization

$$\frac{\partial J(\theta)}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})$$

$$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} + \frac{\lambda}{m} \theta_j \quad \text{for } j \geq 1$$

#### Effect of Regularization

- $\lambda = 0$: No regularization (standard linear regression)
- $\lambda \to \infty$: Heavily penalizes coefficients (underfitting)
- Optimal $\lambda$: Balance between fit and generalization

#### Bias-Variance Tradeoff

```
         Error
          ▲
          │     Total Error
          │    /‾‾‾‾‾‾‾‾
       B  │   /Variance
       i  │  / ‾‾‾‾‾‾
       a  │ /
       s  │/‾‾‾‾‾ Bias²
          │___________→ λ
          0
```

---

## Evaluation Metrics

The implementation provides several metrics to evaluate model performance:

### 1. Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2$$

- Units: Squared units of target variable
- Sensitive to outliers
- **Lower is better**

### 2. Root Mean Squared Error (RMSE)

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2}$$

- Same units as target variable
- More interpretable than MSE
- **Lower is better**

### 3. Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} |y^{(i)} - \hat{y}^{(i)}|$$

- Robust to outliers
- Average absolute deviation
- **Lower is better**

### 4. R² Score (Coefficient of Determination)

$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2}{\sum_{i=1}^{m} (y^{(i)} - \bar{y})^2}$$

Where:
- $\text{SS}_{\text{res}}$ = residual sum of squares
- $\text{SS}_{\text{tot}}$ = total sum of squares
- $\bar{y}$ = mean of target values

- **Range**: $[-\infty, 1]$
- **Interpretation**:
  - $R^2 = 1$: Perfect fit
  - $R^2 = 0$: Model explains 0% variance
  - $R^2 < 0$: Model worse than horizontal line
- **Higher is better**

---

## API Reference

### Constructor

```cpp
LinearRegression();                    // Default: lr=0.01, maxIter=1000
LinearRegression(double lr, size_t maxIter);
LinearRegression(double lr, size_t maxIter, double lambda, bool useReg);
```

### Training Methods

```cpp
// Normal Equation: θ = (X^T*X)^(-1)*X^T*y
void trainNormalEquation(const Matrix& X, const std::vector<double>& y);

// Batch Gradient Descent
void trainGradientDescent(const Matrix& X, const std::vector<double>& y);

// Stochastic Gradient Descent (one sample per update)
void trainSGD(const Matrix& X, const std::vector<double>& y);

// Mini-batch Gradient Descent
void trainMiniBatchGD(const Matrix& X, const std::vector<double>& y, size_t batchSize);
```

### Prediction Methods

```cpp
std::vector<double> predict(const Matrix& X) const;
double predictSingle(const std::vector<double>& x) const;
```

### Evaluation Methods

```cpp
double mse(const Matrix& X, const std::vector<double>& y) const;
double rmse(const Matrix& X, const std::vector<double>& y) const;
double mae(const Matrix& X, const std::vector<double>& y) const;
double r2Score(const Matrix& X, const std::vector<double>& y) const;
```

### Utility Methods

```cpp
void printCoefficients() const;
void printMetrics(const Matrix& X, const std::vector<double>& y) const;
std::vector<double> getTrainingLoss() const;
```

---

## Usage Examples

### Example 1: Simple Linear Regression with Gradient Descent

```cpp
#include "LinearRegression.hpp"
#include <iostream>

int main() {
    // Create training data: y = 2*x + 1 + noise
    Matrix X(10, 1);
    std::vector<double> y;
    
    for (int i = 0; i < 10; ++i) {
        X(i, 0) = i;
        y.push_back(2.0 * i + 1.0);
    }
    
    // Create and train model
    LinearRegression lr(0.01, 1000);
    lr.trainGradientDescent(X, y);
    
    // Print results
    lr.printCoefficients();
    lr.printMetrics(X, y);
    
    // Make predictions
    Matrix testX(5, 1);
    for (int i = 0; i < 5; ++i) testX(i, 0) = i + 10;
    auto predictions = lr.predict(testX);
    
    std::cout << "\nPredictions for x=[10,11,12,13,14]:" << std::endl;
    for (auto pred : predictions) {
        std::cout << pred << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
```

### Example 2: Multiple Features with Regularization

```cpp
LinearRegression lr(0.01, 1000, 0.1, true);  // λ=0.1, regularization enabled
lr.trainGradientDescent(X, y);
```

### Example 3: Stochastic Gradient Descent

```cpp
LinearRegression lr(0.01, 1000);
lr.trainSGD(X, y);  // Online learning with single samples
```

### Example 4: Mini-Batch Training

```cpp
LinearRegression lr(0.01, 1000);
lr.trainMiniBatchGD(X, y, 32);  // Batch size of 32
```

---

## Building the Library

```bash
make              # Build the library
make debug        # Build with debug symbols
make clean        # Remove build artifacts
make rebuild      # Clean and rebuild
make help         # Show available targets
```

---

## Dependencies

- **Matrix Library**: Matrix operations (multiplication, transpose, etc.)
- **Stats Library**: Statistical functions (mean, standard deviation)

---

## Implementation Notes

### Numerical Stability

1. **Feature Scaling**: Features should ideally be normalized/standardized
   - Improves convergence speed
   - Prevents numerical overflow/underflow

2. **Learning Rate Selection**:
   - Too high: Divergence or oscillation
   - Too low: Slow convergence
   - Typical range: [0.001, 0.1]

3. **Convergence**: Monitor training loss to detect convergence

### Time Complexity Summary

| Method | Time per Iteration | Space |
|--------|-------------------|-------|
| Normal Equation | $O(n^3)$ (one-time) | $O(n^2)$ |
| Batch GD | $O(mn)$ | $O(mn)$ |
| SGD | $O(n)$ | $O(n)$ |
| Mini-Batch GD | $O(Bn)$ | $O(Bn)$ |

---

## References

1. **Andrew Ng's Machine Learning Course**: Foundational concepts
2. **Hastie, Tibshirani, Friedman - "The Elements of Statistical Learning"**: Theoretical foundation
3. **Pattern Recognition and Machine Learning (Bishop)**: Advanced topics

---

**Author**: Linear Regression Implementation  
**Version**: 1.0  
**Last Updated**: 2025

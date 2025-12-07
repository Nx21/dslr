#include "../include/Stats.hpp"

double Stats::covariance(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.empty() || y.empty()) {
        throw std::invalid_argument("Data vectors cannot be empty");
    }
    if (x.size() != y.size()) {
        throw std::invalid_argument("Data vectors must have the same size");
    }
    
    double mean_x = mean(x);
    double mean_y = mean(y);
    
    double sum_product = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum_product += (x[i] - mean_x) * (y[i] - mean_y);
    }
    
    return sum_product / x.size();
}

double Stats::correlation(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.empty() || y.empty()) {
        throw std::invalid_argument("Data vectors cannot be empty");
    }
    if (x.size() != y.size()) {
        throw std::invalid_argument("Data vectors must have the same size");
    }
    
    double cov = covariance(x, y);
    double std_x = stdDev(x);
    double std_y = stdDev(y);
    
    if (std_x == 0.0 || std_y == 0.0) {
        throw std::domain_error("Standard deviation cannot be zero");
    }
    
    return cov / (std_x * std_y);
}

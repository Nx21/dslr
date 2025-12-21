#include "../include/StatsCuda.cuh"
#include <algorithm>
#include <numeric>
#include <stdexcept>

StatsCuda::StatsCuda() {
    // Constructor implementation
}

StatsCuda::~StatsCuda() {
    // Destructor implementation
}

double StatsCuda::mean(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double StatsCuda::median(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    size_t n = sorted_data.size();
    if (n % 2 == 0) {
        return (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / 2.0;
    } else {
        return sorted_data[n / 2];
    }
}

double StatsCuda::variance(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    double mean_val = mean(data);
    double sum_sq_diff = 0.0;
    
    for (double val : data) {
        sum_sq_diff += (val - mean_val) * (val - mean_val);
    }
    
    return sum_sq_diff / data.size();
}

double StatsCuda::stdDev(const std::vector<double>& data) {
    return std::sqrt(variance(data));
}

double StatsCuda::min(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    return *std::min_element(data.begin(), data.end());
}

double StatsCuda::max(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    return *std::max_element(data.begin(), data.end());
}

double StatsCuda::range(const std::vector<double>& data) {
    return max(data) - min(data);
}

double StatsCuda::quartile(const std::vector<double>& data, int q) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (q < 0 || q > 4) {
        throw std::invalid_argument("Quartile must be between 0 and 4");
    }
    
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    if (q == 0) return sorted_data.front();
    if (q == 4) return sorted_data.back();
    
    double position = (q / 4.0) * (sorted_data.size() - 1);
    int lower = static_cast<int>(position);
    int upper = lower + 1;
    double fraction = position - lower;
    
    if (upper >= static_cast<int>(sorted_data.size())) {
        return sorted_data[lower];
    }
    
    return sorted_data[lower] * (1.0 - fraction) + sorted_data[upper] * fraction;
}

double StatsCuda::percentile(const std::vector<double>& data, double p) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (p < 0.0 || p > 100.0) {
        throw std::invalid_argument("Percentile must be between 0 and 100");
    }
    
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    double position = (p / 100.0) * (sorted_data.size() - 1);
    int lower = static_cast<int>(position);
    int upper = lower + 1;
    double fraction = position - lower;
    
    if (upper >= static_cast<int>(sorted_data.size())) {
        return sorted_data[lower];
    }
    
    return sorted_data[lower] * (1.0 - fraction) + sorted_data[upper] * fraction;
}

double StatsCuda::covariance(const std::vector<double>& x, const std::vector<double>& y) {
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

double StatsCuda::correlation(const std::vector<double>& x, const std::vector<double>& y) {
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

std::vector<double> StatsCuda::normalize(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    double min_val = min(data);
    double max_val = max(data);
    double range_val = max_val - min_val;
    
    if (range_val == 0.0) {
        throw std::domain_error("Range cannot be zero (all values are identical)");
    }
    
    std::vector<double> normalized(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        normalized[i] = (data[i] - min_val) / range_val;
    }
    
    return normalized;
}

std::vector<double> StatsCuda::standardize(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    double mean_val = mean(data);
    double std_val = stdDev(data);
    
    if (std_val == 0.0) {
        throw std::domain_error("Standard deviation cannot be zero");
    }
    
    std::vector<double> standardized(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        standardized[i] = (data[i] - mean_val) / std_val;
    }
    
    return standardized;
}

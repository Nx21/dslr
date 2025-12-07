#include "../include/Stats.hpp"

std::vector<double> Stats::normalize(const std::vector<double>& data) {
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

std::vector<double> Stats::standardize(const std::vector<double>& data) {
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

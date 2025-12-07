#include "../include/Stats.hpp"

double Stats::mean(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double Stats::median(const std::vector<double>& data) {
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

double Stats::mode(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    double mode_val = sorted_data[0];
    int max_count = 1;
    int current_count = 1;
    
    for (size_t i = 1; i < sorted_data.size(); ++i) {
        if (sorted_data[i] == sorted_data[i - 1]) {
            ++current_count;
            if (current_count > max_count) {
                max_count = current_count;
                mode_val = sorted_data[i];
            }
        } else {
            current_count = 1;
        }
    }
    
    return mode_val;
}

double Stats::variance(const std::vector<double>& data) {
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

double Stats::stdDev(const std::vector<double>& data) {
    return std::sqrt(variance(data));
}

double Stats::min(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    return *std::min_element(data.begin(), data.end());
}

double Stats::max(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    return *std::max_element(data.begin(), data.end());
}

double Stats::range(const std::vector<double>& data) {
    return max(data) - min(data);
}

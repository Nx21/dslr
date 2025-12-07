#include "../include/Stats.hpp"

double Stats::quartile(const std::vector<double>& data, int q) {
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

double Stats::percentile(const std::vector<double>& data, double p) {
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

#ifndef STATS_CUDA_CUH
#define STATS_CUDA_CUH

#include <vector>
#include <cmath>

class StatsCuda {
public:
    // Constructor
    StatsCuda();
    ~StatsCuda();

    // Descriptive statistics
    static double mean(const std::vector<double>& data);
    static double median(const std::vector<double>& data);
    static double variance(const std::vector<double>& data);
    static double stdDev(const std::vector<double>& data);
    static double min(const std::vector<double>& data);
    static double max(const std::vector<double>& data);
    static double range(const std::vector<double>& data);

    // CUDA accelerated operations
    static double meanGPU(const std::vector<double>& data);
    static double sumGPU(const std::vector<double>& data);
    static double varianceGPU(const std::vector<double>& data);
    static double stdDevGPU(const std::vector<double>& data);

    // Quartile and percentile functions
    static double quartile(const std::vector<double>& data, int q);
    static double percentile(const std::vector<double>& data, double p);

    // Correlation and covariance
    static double covariance(const std::vector<double>& x, const std::vector<double>& y);
    static double correlation(const std::vector<double>& x, const std::vector<double>& y);

    // Normalization and standardization
    static std::vector<double> normalize(const std::vector<double>& data);
    static std::vector<double> standardize(const std::vector<double>& data);
};

#endif // STATS_CUDA_CUH

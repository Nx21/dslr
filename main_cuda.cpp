#include <iostream>
#include <vector>
#include <iomanip>

#include "MatrixCuda.cuh"
#include "StatsCuda.cuh"

void printVector(const std::vector<double>& vec, const std::string& label) {
    std::cout << label << ": [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void testStatsCuda() {
    std::cout << "\n=== Testing StatsCuda Library ===" << std::endl;

    std::vector<double> data = {12.5, 15.3, 18.7, 14.2, 16.8, 19.1, 13.4, 17.5, 20.2, 15.9};
    printVector(data, "Data");

    std::cout << "\nDescriptive Statistics:" << std::endl;
    std::cout << "  Mean:     " << std::fixed << std::setprecision(4) << StatsCuda::mean(data) << std::endl;
    std::cout << "  Median:   " << StatsCuda::median(data) << std::endl;
    std::cout << "  Std Dev:  " << StatsCuda::stdDev(data) << std::endl;
    std::cout << "  Variance: " << StatsCuda::variance(data) << std::endl;
    std::cout << "  Min:      " << StatsCuda::min(data) << std::endl;
    std::cout << "  Max:      " << StatsCuda::max(data) << std::endl;
    std::cout << "  Range:    " << StatsCuda::range(data) << std::endl;

    std::cout << "\nQuartiles:" << std::endl;
    std::cout << "  Q1 (25%): " << StatsCuda::quartile(data, 1) << std::endl;
    std::cout << "  Q2 (50%): " << StatsCuda::quartile(data, 2) << std::endl;
    std::cout << "  Q3 (75%): " << StatsCuda::quartile(data, 3) << std::endl;

    std::vector<double> normalized = StatsCuda::normalize(data);
    printVector(normalized, "\nNormalized");

    std::vector<double> standardized = StatsCuda::standardize(data);
    printVector(standardized, "Standardized");

    std::vector<double> data2 = {10.0, 12.5, 15.0, 13.2, 14.8, 16.5, 11.8, 15.2, 17.8, 14.1};
    std::cout << "\nCorrelation with second dataset: " << StatsCuda::correlation(data, data2) << std::endl;
    std::cout << "Covariance with second dataset: " << StatsCuda::covariance(data, data2) << std::endl;
}

void testMatrixCuda() {
    std::cout << "\n=== Testing MatrixCuda Library ===" << std::endl;

    MatrixCuda A = {{1.0, 2.0, 3.0},
                    {4.0, 5.0, 6.0},
                    {7.0, 8.0, 9.0}};

    MatrixCuda B = {{9.0, 8.0, 7.0},
                    {6.0, 5.0, 4.0},
                    {3.0, 2.0, 1.0}};

    std::cout << "\nMatrix A:" << std::endl;
    A.print();

    std::cout << "\nMatrix B:" << std::endl;
    B.print();

    std::cout << "\nMatrix A + B:" << std::endl;
    MatrixCuda C = A + B;
    C.print();

    std::cout << "\nMatrix A * B:" << std::endl;
    MatrixCuda D = A * B;
    D.print();

    std::cout << "\nMatrix A * 2:" << std::endl;
    MatrixCuda E = A * 2.0;
    E.print();

    std::cout << "\nMatrix A properties:" << std::endl;
    std::cout << "  Rows: " << A.getRows() << std::endl;
    std::cout << "  Cols: " << A.getCols() << std::endl;

    std::cout << "\nTranspose of A:" << std::endl;
    MatrixCuda AT = A.transpose();
    AT.print();
}

void testCombinedCuda() {
    std::cout << "\n=== Testing Combined MatrixCuda & StatsCuda ===" << std::endl;

    MatrixCuda data = {{12.5, 15.3, 18.7},
                       {14.2, 16.8, 19.1},
                       {13.4, 17.5, 20.2},
                       {15.9, 14.6, 16.3},
                       {17.1, 18.9, 15.8}};

    std::cout << "\nData Matrix:" << std::endl;
    data.print();

    std::vector<double> col1, col2, col3;
    for (size_t i = 0; i < 5; ++i) {
        col1.push_back(data(i, 0));
        col2.push_back(data(i, 1));
        col3.push_back(data(i, 2));
    }

    std::cout << "\nColumn-wise statistics:" << std::endl;
    std::cout << "  Column 1 - Mean: " << std::fixed << std::setprecision(4)
              << StatsCuda::mean(col1) << ", Std: " << StatsCuda::stdDev(col1) << std::endl;
    std::cout << "  Column 2 - Mean: " << StatsCuda::mean(col2)
              << ", Std: " << StatsCuda::stdDev(col2) << std::endl;
    std::cout << "  Column 3 - Mean: " << StatsCuda::mean(col3)
              << ", Std: " << StatsCuda::stdDev(col3) << std::endl;

    std::cout << "\nCorrelations between columns:" << std::endl;
    std::cout << "  Col1 & Col2: " << StatsCuda::correlation(col1, col2) << std::endl;
    std::cout << "  Col1 & Col3: " << StatsCuda::correlation(col1, col3) << std::endl;
    std::cout << "  Col2 & Col3: " << StatsCuda::correlation(col2, col3) << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  DSLR CUDA Library Test Program" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        testStatsCuda();
        testMatrixCuda();
        testCombinedCuda();

        std::cout << "\n========================================" << std::endl;
        std::cout << "  All CUDA tests completed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

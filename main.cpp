#include <iostream>
#include <vector>
#include <iomanip>

// C++ Libraries
#include "Matrix.hpp"
#include "Stats.hpp"

void printVector(const std::vector<double>& vec, const std::string& label) {
    std::cout << label << ": [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void testStatsLibrary() {
    std::cout << "\n=== Testing Stats Library ===" << std::endl;
    
    // Sample data
    std::vector<double> data = {12.5, 15.3, 18.7, 14.2, 16.8, 19.1, 13.4, 17.5, 20.2, 15.9};
    
    printVector(data, "Data");
    
    // Descriptive statistics
    std::cout << "\nDescriptive Statistics:" << std::endl;
    std::cout << "  Mean:     " << std::fixed << std::setprecision(4) << Stats::mean(data) << std::endl;
    std::cout << "  Median:   " << Stats::median(data) << std::endl;
    std::cout << "  Std Dev:  " << Stats::stdDev(data) << std::endl;
    std::cout << "  Variance: " << Stats::variance(data) << std::endl;
    std::cout << "  Min:      " << Stats::min(data) << std::endl;
    std::cout << "  Max:      " << Stats::max(data) << std::endl;
    std::cout << "  Range:    " << Stats::range(data) << std::endl;
    
    // Quartiles
    std::cout << "\nQuartiles:" << std::endl;
    std::cout << "  Q1 (25%): " << Stats::quartile(data, 1) << std::endl;
    std::cout << "  Q2 (50%): " << Stats::quartile(data, 2) << std::endl;
    std::cout << "  Q3 (75%): " << Stats::quartile(data, 3) << std::endl;
    
    // Normalization
    std::vector<double> normalized = Stats::normalize(data);
    printVector(normalized, "\nNormalized");
    
    // Standardization
    std::vector<double> standardized = Stats::standardize(data);
    printVector(standardized, "Standardized");
    
    // Correlation
    std::vector<double> data2 = {10.0, 12.5, 15.0, 13.2, 14.8, 16.5, 11.8, 15.2, 17.8, 14.1};
    std::cout << "\nCorrelation with second dataset: " 
              << Stats::correlation(data, data2) << std::endl;
    std::cout << "Covariance with second dataset: " 
              << Stats::covariance(data, data2) << std::endl;
}

void testMatrixLibrary() {
    std::cout << "\n=== Testing Matrix Library ===" << std::endl;
    
    // Create matrices
    Matrix A(3, 3);
    A(0, 0) = 1.0; A(0, 1) = 2.0; A(0, 2) = 3.0;
    A(1, 0) = 4.0; A(1, 1) = 5.0; A(1, 2) = 6.0;
    A(2, 0) = 7.0; A(2, 1) = 8.0; A(2, 2) = 9.0;
    
    Matrix B(3, 3);
    B(0, 0) = 9.0; B(0, 1) = 8.0; B(0, 2) = 7.0;
    B(1, 0) = 6.0; B(1, 1) = 5.0; B(1, 2) = 4.0;
    B(2, 0) = 3.0; B(2, 1) = 2.0; B(2, 2) = 1.0;
    
    std::cout << "\nMatrix A:" << std::endl;
    A.print();
    
    std::cout << "\nMatrix B:" << std::endl;
    B.print();
    
    // Matrix operations
    std::cout << "\nMatrix A + B:" << std::endl;
    Matrix C = A + B;
    C.print();
    
    std::cout << "\nMatrix A * B:" << std::endl;
    Matrix D = A * B;
    D.print();
    
    std::cout << "\nMatrix A * 2:" << std::endl;
    Matrix E = A * 2.0;
    E.print();
    
    // Matrix properties
    std::cout << "\nMatrix A properties:" << std::endl;
    std::cout << "  Rows: " << A.getRows() << std::endl;
    std::cout << "  Cols: " << A.getCols() << std::endl;
    
    // Transpose
    std::cout << "\nTranspose of A:" << std::endl;
    Matrix AT = A.transpose();
    AT.print();
}

void testCombinedOperations() {
    std::cout << "\n=== Testing Combined Matrix & Stats ===" << std::endl;
    
    // Create a matrix with sample data
    Matrix data(5, 3);
    data(0, 0) = 12.5; data(0, 1) = 15.3; data(0, 2) = 18.7;
    data(1, 0) = 14.2; data(1, 1) = 16.8; data(1, 2) = 19.1;
    data(2, 0) = 13.4; data(2, 1) = 17.5; data(2, 2) = 20.2;
    data(3, 0) = 15.9; data(3, 1) = 14.6; data(3, 2) = 16.3;
    data(4, 0) = 17.1; data(4, 1) = 18.9; data(4, 2) = 15.8;
    
    std::cout << "\nData Matrix:" << std::endl;
    data.print();
    
    // Extract columns for statistical analysis
    std::vector<double> col1, col2, col3;
    for (size_t i = 0; i < 5; ++i) {
        col1.push_back(data(i, 0));
        col2.push_back(data(i, 1));
        col3.push_back(data(i, 2));
    }
    
    std::cout << "\nColumn-wise statistics:" << std::endl;
    std::cout << "  Column 1 - Mean: " << std::fixed << std::setprecision(4) 
              << Stats::mean(col1) << ", Std: " << Stats::stdDev(col1) << std::endl;
    std::cout << "  Column 2 - Mean: " << Stats::mean(col2) 
              << ", Std: " << Stats::stdDev(col2) << std::endl;
    std::cout << "  Column 3 - Mean: " << Stats::mean(col3) 
              << ", Std: " << Stats::stdDev(col3) << std::endl;
    
    std::cout << "\nCorrelations between columns:" << std::endl;
    std::cout << "  Col1 & Col2: " << Stats::correlation(col1, col2) << std::endl;
    std::cout << "  Col1 & Col3: " << Stats::correlation(col1, col3) << std::endl;
    std::cout << "  Col2 & Col3: " << Stats::correlation(col2, col3) << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  DSLR Library Test Program" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        // Test Stats library
        testStatsLibrary();
        
        // Test Matrix library
        testMatrixLibrary();
        
        // Test combined operations
        testCombinedOperations();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "  All tests completed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
   
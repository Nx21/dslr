#include "MatrixCuda.cuh"
#include <iostream>
#include <vector>

int main() {
    try {
        std::cout << "Testing CUDA Matrix library...\n";
        
        // Check CUDA device
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            std::cout << "No CUDA devices found. Running basic tests only.\n";
        } else {
            std::cout << "Found " << deviceCount << " CUDA device(s).\n";
        }
        
        // Test constructors
        MatrixCuda m1(3, 3, 1.0);  // 3x3 matrix filled with 1.0
        MatrixCuda m2 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};  // Initialize with values
        
        std::cout << "\nMatrix m1 (3x3 filled with 1.0):\n";
        m1.print();
        
        std::cout << "\nMatrix m2 (initialized with values):\n";
        m2.print();
        
        // Test operations
        MatrixCuda m3 = m1 + m2;
        std::cout << "\nMatrix m1 + m2:\n";
        m3.print();
        
        // Test scalar operations
        MatrixCuda m4 = m2 * 2.0;
        std::cout << "\nMatrix m2 * 2.0:\n";
        m4.print();
        
        // Test static methods
        MatrixCuda identity = MatrixCuda::identity(3);
        std::cout << "\n3x3 Identity matrix:\n";
        identity.print();
        
        // Test transpose
        MatrixCuda m2_T = m2.transpose();
        std::cout << "\nTranspose of m2:\n";
        m2_T.print();
        
        // Test vector operations
        std::vector<double> vec = {1.0, 2.0, 3.0};
        MatrixCuda vecMatrix = MatrixCuda::fromVector(vec, true); // Column vector
        std::cout << "\nColumn vector from std::vector {1, 2, 3}:\n";
        vecMatrix.print();
        
        std::cout << "\nCUDA Matrix library test completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
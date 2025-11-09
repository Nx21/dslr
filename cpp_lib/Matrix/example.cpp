#include "Matrix.hpp"
#include <iostream>

int main() {
    // Test basic Matrix functionality
    std::cout << "Testing Matrix library...\n";
    
    // Test constructors
    Matrix m1(3, 3, 1.0);  // 3x3 matrix filled with 1.0
    Matrix m2 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};  // Initialize with values
    
    std::cout << "Matrix m1 (3x3 filled with 1.0):\n";
    m1.print();
    
    std::cout << "\nMatrix m2 (initialized with values):\n";
    m2.print();
    
    // Test operations
    Matrix m3 = m1 + m2;
    std::cout << "\nMatrix m1 + m2:\n";
    m3.print();
    
    // Test static methods
    Matrix identity = Matrix::identity(3);
    std::cout << "\n3x3 Identity matrix:\n";
    identity.print();
    
    // Test transpose
    Matrix m2_T = m2.transpose();
    std::cout << "\nTranspose of m2:\n";
    m2_T.print();
    
    std::cout << "\nMatrix library test completed successfully!\n";
    return 0;
}
#include "cais/solver.hpp"
#include <iostream>

void print_matrix(const std::string& title, const cais::Matrix& mat) {
    std::cout << "--- " << title << " (" << mat.rows() << "x" << mat.cols() << ") ---\n"
              << mat << "\n" << std::endl;
}

void print_vector(const std::string& title, const cais::Vector& vec) {
    std::cout << "--- " << title << " (" << vec.size() << ") ---\n"
              << vec << "\n" << std::endl;
}

int main() {
    try {
        // ====================================================================
        // Test 1: Matrix Multiplication (matmul)
        // ====================================================================

        // 1. Prepare test data for matmul
        cais::Matrix A(2, 3); // 2x3 matrix
        A << 1.0f, 2.0f, 3.0f,
             4.0f, 5.0f, 6.0f;

        cais::Matrix B(3, 4); // 3x4 matrix
        B << 7.0f, 8.0f, 9.0f, 10.0f,
             11.0f, 12.0f, 13.0f, 14.0f,
             15.0f, 16.0f, 17.0f, 18.0f;
        
        cais::Matrix C_cpu, C_gpu;

        print_matrix("Input Matrix A", A);
        print_matrix("Input Matrix B", B);

        // 2. Run on CPU
        std::cout << "\n--- Running on CPU ---\n";
        cais::Solver cpu_solver(cais::ExecutionMode::CPU);
        cpu_solver.matmul(A, B, C_cpu);
        print_matrix("CPU Result", C_cpu);

        // 3. Run on GPU
        std::cout << "--- Running on GPU ---\n";
        cais::Solver gpu_solver(cais::ExecutionMode::GPU);
        gpu_solver.matmul(A, B, C_gpu);
        print_matrix("GPU Result", C_gpu);

        // 4. Matmul Verification
        if (C_cpu.isApprox(C_gpu)) {
            std::cout << ">>> MATMUL VERIFICATION: SUCCESS! Results are identical.\n\n";
        } else {
            std::cout << ">>> MATMUL VERIFICATION: FAILURE! Results are different.\n\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "A fatal error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

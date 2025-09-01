#include "cais/solver.hpp"
#include <iostream>
#include <chrono>

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


    // 1. Prepare large random test data for matmul
    const int rowsA = 1000, colsA = 1000, colsB = 1000;
    cais::Matrix A = cais::Matrix::Random(rowsA, colsA);
    cais::Matrix B = cais::Matrix::Random(colsA, colsB);
    cais::Matrix C_cpu, C_gpu;

    std::cout << "Input Matrix A: " << A.rows() << "x" << A.cols() << std::endl;
    std::cout << "Input Matrix B: " << B.rows() << "x" << B.cols() << std::endl;


    // 2. Run on CPU and time it
    std::cout << "\n--- Running on CPU ---\n";
    cais::Solver cpu_solver(cais::ExecutionMode::CPU);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_solver.matmul(A, B, C_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(cpu_end - cpu_start).count();
    std::cout << "CPU Result: " << C_cpu.rows() << "x" << C_cpu.cols() << std::endl;
    std::cout << "CPU matmul time: " << cpu_time << " seconds" << std::endl;

    // 3. Run on GPU and time it
    std::cout << "--- Running on GPU ---\n";
    cais::Solver gpu_solver(cais::ExecutionMode::GPU);
    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu_solver.matmul(A, B, C_gpu);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double>(gpu_end - gpu_start).count();
    std::cout << "GPU Result: " << C_gpu.rows() << "x" << C_gpu.cols() << std::endl;
    std::cout << "GPU matmul time: " << gpu_time << " seconds" << std::endl;

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

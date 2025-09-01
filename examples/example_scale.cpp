#include <iostream>
#include "cais/solver.hpp"
#include <chrono>

void print_matrix(const std::string& title, const cais::Matrix& mat) {
    std::cout << "--- " << title << " (" << mat.rows() << "x" << mat.cols() << ") ---\n"
              << mat << "\n" << std::endl;
}

int main() {
    try {
        std::cout << "========================================\n";
        std::cout << "      EXAMPLE: SCALE (A = A * s)        \n";
        std::cout << "========================================\n\n";

        // Use a large random matrix
        const int rows = 1000, cols = 1000;
        cais::Matrix A_cpu = cais::Matrix::Random(rows, cols);
        cais::Matrix A_gpu = A_cpu;
        const float scalar = 10.0f;

        std::cout << "Original Matrix: " << A_cpu.rows() << "x" << A_cpu.cols() << std::endl;
        std::cout << "Scalar: " << scalar << std::endl;

        // --- CPU execution with timing ---
        cais::Solver cpu_solver(cais::ExecutionMode::CPU);
        auto cpu_start = std::chrono::high_resolution_clock::now();
        cpu_solver.scale(A_cpu, scalar);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double>(cpu_end - cpu_start).count();
        std::cout << "CPU Result: " << A_cpu.rows() << "x" << A_cpu.cols() << std::endl;
        std::cout << "CPU Time: " << cpu_time << " seconds" << std::endl;

        // --- GPU execution with timing ---
        cais::Solver gpu_solver(cais::ExecutionMode::GPU);
        auto gpu_start = std::chrono::high_resolution_clock::now();
        gpu_solver.scale(A_gpu, scalar);
        auto gpu_end = std::chrono::high_resolution_clock::now();
        double gpu_time = std::chrono::duration<double>(gpu_end - gpu_start).count();
        std::cout << "GPU Result: " << A_gpu.rows() << "x" << A_gpu.cols() << std::endl;
        std::cout << "GPU Time: " << gpu_time << " seconds" << std::endl;

        // --- Verification ---
        if (A_cpu.isApprox(A_gpu)) {
            std::cout << ">>> SCALE VERIFICATION: SUCCESS! Results are identical.\n";
        } else {
            std::cout << ">>> SCALE VERIFICATION: FAILURE! Results are different.\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "A fatal error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

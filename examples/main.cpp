#include "cais/solver.hpp" // A API pública da sua biblioteca
#include <iostream>

// --- Funções Auxiliares para Impressão ---

// Imprime uma matriz Eigen de forma clara e legível.
void print_matrix(const std::string& title, const cais::Matrix& mat) {
    std::cout << "--- " << title << " (" << mat.rows() << "x" << mat.cols() << ") ---\n"
              << mat << "\n" << std::endl;
}

// Imprime um vetor Eigen.
void print_vector(const std::string& title, const cais::Vector& vec) {
    std::cout << "--- " << title << " (" << vec.size() << ") ---\n"
              << vec << "\n" << std::endl;
}


int main() {
    try {
        // ====================================================================
        // Teste 1: Multiplicação de Matrizes (matmul)
        // ====================================================================

        // 1. Preparar dados de teste para matmul
        cais::Matrix A(2, 3); // Matriz 2x3
        A << 1.0f, 2.0f, 3.0f,
             4.0f, 5.0f, 6.0f;

        cais::Matrix B(3, 4); // Matriz 3x4
        B << 7.0f, 8.0f, 9.0f, 10.0f,
             11.0f, 12.0f, 13.0f, 14.0f,
             15.0f, 16.0f, 17.0f, 18.0f;
        
        cais::Matrix C_cpu, C_gpu;

        print_matrix("Matriz de Entrada A", A);
        print_matrix("Matriz de Entrada B", B);

        // 2. Executar na CPU
        std::cout << "\n--- Rodando na CPU ---\n";
        cais::Solver cpu_solver(cais::ExecutionMode::CPU);
        cpu_solver.matmul(A, B, C_cpu);
        print_matrix("Resultado da CPU", C_cpu);

        // 3. Executar na GPU
        std::cout << "--- Rodando na GPU ---\n";
        cais::Solver gpu_solver(cais::ExecutionMode::GPU);
        gpu_solver.matmul(A, B, C_gpu);
        print_matrix("Resultado da GPU", C_gpu);

        // 4. Verificação de Matmul
        if (C_cpu.isApprox(C_gpu)) {
            std::cout << ">>> VERIFICACAO MATMUL: SUCESSO! Resultados sao identicos.\n\n";
        } else {
            std::cout << ">>> VERIFICACAO MATMUL: FALHA! Resultados sao diferentes.\n\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Ocorreu um erro fatal: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


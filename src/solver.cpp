#include "cais/solver.hpp"
#include "cpu_engine.hpp"
#include "opencl_engine.hpp"

#include <iostream>
#include <stdexcept>

namespace cais {

Solver::Solver(ExecutionMode mode) {
    if (mode == ExecutionMode::GPU) {
        try {
            m_engine = std::make_unique<OpenClEngine>();
            std::cout << "[CAIS Solver] -> Motor GPU (OpenCL) inicializado com sucesso." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[CAIS Solver] -> AVISO: Falha ao inicializar o motor GPU: " << e.what() << std::endl;
            std::cerr << "[CAIS Solver] -> Usando o motor de CPU como fallback." << std::endl;
            m_engine = std::make_unique<CpuEngine>();
        }
    } else {
        m_engine = std::make_unique<CpuEngine>();
        std::cout << "[CAIS Solver] -> Motor CPU (Eigen) inicializado com sucesso." << std::endl;
    }
}

void Solver::matmul(const Matrix& A, const Matrix& B, Matrix& C)
{
    m_engine->matmul(A, B, C);
}

void Solver::add(const Matrix& A, const Matrix& B, Matrix& C)
{
    m_engine->add(A, B, C);
}

void Solver::scale(Matrix& A, const float scalar_value)
{
    m_engine->scale(A, scalar_value);
}

} // namespace cais
#pragma once
#include "cais/engine.hpp" // Note a extensão .hpp
#include "cais/common.hpp"
#include <memory>

namespace cais {

class Solver {
public:
    explicit Solver(ExecutionMode mode = ExecutionMode::CPU);

    // --- NOVA FUNÇÃO PÚBLICA ---
    void matmul(
        const Matrix& A, const Matrix& B, Matrix& C);

    // Placeholder for matrix_vector_multiply_add
    void matrix_vector_multiply_add(const Matrix& A, const Vector& x, const Vector& y, Vector& result);

private:
    std::unique_ptr<Engine> m_engine;
};

} // namespace cais

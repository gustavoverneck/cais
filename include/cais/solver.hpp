#pragma once
#include "cais/engine.hpp" // Note a extensão .hpp
#include "cais/common.hpp"
#include <memory>

namespace cais {

class Solver {
public:
    explicit Solver(ExecutionMode mode = ExecutionMode::CPU);

    void matmul(
        const Matrix& A, 
        const Matrix& B, 
        Matrix& C
    );

    void add(
        const Matrix& A,
        const Matrix& B,
        Matrix& C
    );
    
    void scale(
        Matrix& A,
        const float scalar_value
    );

private:
    std::unique_ptr<Engine> m_engine;
};

} // namespace cais

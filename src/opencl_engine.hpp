#pragma once

#include "cais/engine.hpp" // Inclui a interface pública
#include "opencl.hpp"      // Inclui o seu wrapper OpenCL

namespace cais {

std::string opencl_c_container();

class OpenClEngine : public Engine {
public:
    OpenClEngine();

    // Matrices Multiplication: C = A * B
    void matmul(
        const Matrix& A, 
        const Matrix& B, 
        Matrix& C
    ) override;

    void add(
        const Matrix& A, 
        const Matrix& B, 
        Matrix& C
    ) override;

    void scale(
        Matrix& A,
        const float scalar_value
    ) override;

private:
    Device m_device;
};

} // namespace cais

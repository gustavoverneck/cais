#include "cpu_engine.hpp"
#include <stdexcept>

namespace cais {

void CpuEngine::matmul(const Matrix& A, const Matrix& B, Matrix& C)
{
    // Check Dimensions
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Inner matrix dimensions must be equal.");
    }

    // Resize C
    C.resize(A.rows(), B.cols());

    // Multiply using Eigen
    C = A * B;
}

void CpuEngine::add(const Matrix& A, const Matrix& B, Matrix& C)
{
    // Check Dimensions
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("Matrix dimensions must be equal for addition.");
    }

    // Resize C
    C.resize(A.rows(), A.cols());

    // Add using Eigen
    C = A + B;
}

} // namespace cais
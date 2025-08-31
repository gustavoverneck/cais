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

} // namespace cais
#pragma once

#include <Eigen/Dense>

namespace cais 
{

using Vector = Eigen::VectorXf;
using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class Engine {
public:
    virtual ~Engine() = default;

    // Add the solvers here
    
    // Matrices Multiplication
    virtual void matmul(
        const Matrix& A,
        const Matrix& B,
        Matrix& C
    ) = 0;

};


} // namespace cais
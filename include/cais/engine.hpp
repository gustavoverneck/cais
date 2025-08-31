#pragma once

#include <Eigen/Dense>

namespace cais 
{

using Vector = Eigen::VectorXf;
using Matrix = Eigen::MatrixXf;

class Engine {
public:
    virtual ~Engine() = default;
};


} // namespace cais
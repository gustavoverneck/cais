#pragma once

#include "cais/engine.hpp"

namespace cais {

class CpuEngine : public Engine {
public:
    void matmul(
        const Matrix& A,
        const Matrix& B,
        Matrix& C) override;

    };


} // namespace cais
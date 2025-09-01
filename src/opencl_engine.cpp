#include "opencl_engine.hpp"
#include <stdexcept>

namespace cais {

// --- Device Definition ---
OpenClEngine::OpenClEngine() {
    m_device = Device(select_device_with_most_flops());
}

// -----------------------------------------------------------------------------------------------

// --- GPU MATMUL ---
void OpenClEngine::matmul(const Matrix& A, const Matrix& B, Matrix& C)
{
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Inner dimensions of matrices must be equal.");
    }

    const uint M = static_cast<uint>(A.rows());
    const uint K = static_cast<uint>(A.cols());
    const uint N = static_cast<uint>(B.cols());
    C.resize(M, N);

    // 1. Allocate memory on the GPU and copy the data from A and B.
    Memory<float> A_dev(m_device, A.size(), 1, const_cast<float*>(A.data()));
    Memory<float> B_dev(m_device, B.size(), 1, const_cast<float*>(B.data()));
    Memory<float> C_dev(m_device, C.size());

    // 2. Create the Kernel object, passing the GPU buffers and dimensions.
    Kernel kernel(m_device, static_cast<ulong>(M) * N, "matmul_kernel",
                  A_dev, B_dev, C_dev, M, N, K);
    
    // 3. Run the kernel.
    kernel.run();

    // 4. Read the result from the GPU back to the host.
    C_dev.read_from_device();

    // 5. Copy the result to the output Eigen matrix.
    memcpy(C.data(), C_dev.data(), C.size() * sizeof(float));
}

// -----------------------------------------------------------------------------------------------

// --- GPU ADD ---
void OpenClEngine::add(const Matrix& A, const Matrix& B, Matrix& C)
{
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("Matrix dimensions must be equal for addition.");
    }

    const ulong N = A.rows();
    // Resize C
    C.resize(N, N);

    // 1. Allocate memory on the GPU and copy the data from A and B.
    Memory<float> A_dev(m_device, A.size(), 1, const_cast<float*>(A.data()));
    Memory<float> B_dev(m_device, B.size(), 1, const_cast<float*>(B.data()));
    Memory<float> C_dev(m_device, C.size());

    // 2. Create the Kernel object, passing the GPU buffers and dimensions.
    Kernel kernel(m_device, static_cast<ulong>(N) * N, "add_kernel",
                  A_dev, B_dev, C_dev);
    
    // 3. Run the kernel.
    kernel.run();

    // 4. Read the result from the GPU back to the host.
    C_dev.read_from_device();

    // 5. Copy the result to the output Eigen matrix.
    memcpy(C.data(), C_dev.data(), C.size() * sizeof(float));
}

// -----------------------------------------------------------------------------------------------



} // namespace cais

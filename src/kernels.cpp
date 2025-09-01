#include "kernel.hpp" // note: unbalanced round brackets () are not allowed and string literals can't be arbitrarily long, so periodically interrupt with )+R(
string opencl_c_container() { return R( // ########################## begin of OpenCL C code ####################################################################



kernel void add_kernel(global float* A, global float* B, global float* C) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	const uint n = get_global_id(0);
	C[n] = A[n]+B[n];
}


kernel void matmul_kernel(global const float* A, global const float* B, global float* C, const uint M, const uint N, const uint K) {
    const int global_id = get_global_id(0);

    const int i = global_id / N; // Integer division gives the row (i).
    const int j = global_id % N; // Remainder gives the column (j).

    // Perform dot product of row 'i' of A with column 'j' of B.
    float sum = 0.0f;
    for (uint k = 0; k < K; ++k) {
        // Access A[i][k] and B[k][j] in linear memory.
        sum += A[i * K + k] * B[k * N + j];
    }

    // Store the result in the correct position C[i][j] in linear memory.
    C[i * N + j] = sum;
}

kernel void scale_kernel(global float* A, const float scalar_value) {
    const uint  n = get_global_id(0);

    A[n] = A[n] * scalar_value;
}


);} // ############################################################### end of OpenCL C code #####################################################################
#include "kernel.hpp" // note: unbalanced round brackets () are not allowed and string literals can't be arbitrarily long, so periodically interrupt with )+R(
string opencl_c_container() { return R( // ########################## begin of OpenCL C code ####################################################################



kernel void add_kernel(global float* A, global float* B, global float* C) { // equivalent to "for(uint n=0u; n<N; n++) {", but executed in parallel
	const uint n = get_global_id(0);
	C[n] = A[n]+B[n];
}


kernel void matrix_multiply_kernel( global const float* A, global const float* B, global float* C, const int widthA, const int widthB
    ) {
        int col = get_global_id(0); // X -> Column
        int row = get_global_id(1); // Y -> Row

        float sum = 0.0f;
        for (int k = 0; k < widthA; ++k) {
            sum += A[row * widthA + k] * B[k * widthB + col];
        }

        C[row * widthB + col] = sum;
    }


);} // ############################################################### end of OpenCL C code #####################################################################
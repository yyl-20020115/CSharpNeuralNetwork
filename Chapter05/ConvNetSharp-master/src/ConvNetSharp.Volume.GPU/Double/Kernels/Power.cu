﻿extern "C" {
    __global__ void Run(int n, double* __restrict left, double* __restrict right, double* __restrict output) {
		int i = blockIdx.x*blockDim.x + threadIdx.x;
		if (i < n) output[i] = pow(left[i], right[i]);
	}
}
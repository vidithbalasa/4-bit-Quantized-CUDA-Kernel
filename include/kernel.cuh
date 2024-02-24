#ifndef KERNEL_H
#define KERNEL_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "global.h"

#define MAX_4BIT 7
#define MIN_4BIT -7

__device__ int8_t quantize(int8_t a, int8_t b);
__global__ void quantize_array_kernel(const int8_t* input, int8_t* output, int n);

#endif // KERNEL_H

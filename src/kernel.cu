#include <kernel.cuh>
#include <iostream>

__device__ int8_t quantize(int8_t a, int8_t b) {
    /* Quantizes a pair of values to fit two 4-bit signed integers into a single byte */
    
    if (a > MAX_4BIT || a < MIN_4BIT || b > MAX_4BIT || b < MIN_4BIT) {
        printf("ERROR: Input integers must fit in 4 bits\n");
        return 0x80; // Set highest bit to indicate error
    }

    // Mask for value of signed ints: 2^3 & 2^8 ( [0]000[0]000 )
    int8_t a_val = (a >> 7) & 1;
    int8_t b_val = (a >> 7) & 1;
    
    // Masking for absolute value
    int8_t a_mask = a >> 7;
    a = (a ^ a_mask) - a_mask;
    int8_t b_mask = b >> 7;
    b = (b ^ b_mask) - b_mask;

    // Transalte 4bit ints to left & right half of 8-bit int
    a_val |= (a & 7) * 16;
    b_val |= b & 7;

    return int8_t(a_val + b_val);
}

__global__ void quantize_array_kernel(const int8_t* input, int8_t* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * idx + 1 < n && idx < (n / 2 + n % 2)) {
        int8_t a = input[2 * idx];
        int8_t b = input[2 * idx + 1];
        output[idx] = quantize(a, b);
    }
}


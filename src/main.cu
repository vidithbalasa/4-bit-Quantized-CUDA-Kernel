#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

int quantize(int a, int b = 0) {
    /*
    quantizes a pair of values to fit two 4-bit signed integers into a single byte
    */
    
    // input ints must fit in 4 bits
    if (a > 7 || a < -7 || b > 7 || b < -7) {
        printf("ERROR: quantize out of range\n");
        exit(1);
    }

    // Signed Ints: 2^3 & 2^8 ( [0]000[0]000 )
    int a_val = (a >> 31) & 128;
    int b_val = (b >> 31) & 8;

    // Transalte a[4-bit] to left half of 8-bit int
    a_val += (a & 1) * 16;
    a_val += (a & 2) * 32;
    a_val += (a & 4) * 64;

    // Transalte b[4-bit] to right half of 8-bit int (last 3 bits)
    b_val += b & 3;

    // Combine a and b into a single byte
    return a_val + b_val;
}

int main() {
    int a = 6;
    int b = -7;
    int c = quantize(a, b);
    printf("c = %d\n", c);

    return 0;
}

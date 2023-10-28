#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

int quantize(int8_t a, int8_t b = 0) {
    /*
    quantizes a pair of values to fit two 4-bit signed integers into a single byte
    */
    
    // input ints must fit in 4 bits
    if (a > 7 || a < -7 || b > 7 || b < -7) {
        printf("ERROR: quantize out of range\n");
        exit(1);
    }

    // Signed Ints: 2^3 & 2^8 ( [0]000[0]000 )
    int a_val = (a >> sizeof(int)-1) & 128;
    int8_t b_val = (b >> sizeof(int)-1) & 8;

    // Only care about values
    a = abs(a);
    b = abs(b);

    // Transalte a[4-bit] to left half of 8-bit int
    a_val += (a & 1) * 16;
    a_val += (a & 2) * 16;
    a_val += (a & 4) * 16;

    // Transalte b[4-bit] to right half of 8-bit int (last 3 bits)
    b_val += b & 7;

    // Combine a and b into a single byte
    return int8_t(a_val + b_val);
}

int main() {
    int8_t a = 6;
    int8_t b = -7;
    int8_t c = quantize(a, b);
    printf("c = %d\n", int(sizeof(c)));

    return 0;
}

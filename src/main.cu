#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

int quantize(int8_t a, int8_t b) {
    /*
    quantizes a pair of values to fit two 4-bit signed integers into a single byte
    */
    
    // input ints must fit in 4 bits
    if (a > 7 || a < -7 || b > 7 || b < -7) {
        printf("ERROR: Out of Quantizable Range\n");
        exit(1);
    }

    // Signed Ints: 2^3 & 2^8 ( [0]000[0]000 )
    int8_t a_val = (a >> sizeof(int)-1) & 128;
    int8_t b_val = (b >> sizeof(int)-1) & 8;

    // Only care about values
    a = abs(a);
    b = abs(b);

    // Transalte 4bit ints to left * right half of 8-bit int
    a_val += (a & 7) * 16;
    b_val += b & 7;

    // Combine a and b into a single byte
    return int8_t(a_val + b_val);
}

int8_t* split_ints(int8_t c, int8_t* split) {
    /*
    splits a single byte into two "4-bit" signed integers
    */

    // Split the byte into two 4-bit integers
    int8_t a = c >> 4;
    int8_t b = c & 15;

    // Get signed bit (4th bit from right)
    int8_t a_sign = a >> 3;
    int8_t a_multiplier = (1 * (1 - a_sign)) + (-1 * a_sign);

    int8_t b_sign = b >> 3;
    int8_t b_multiplier = (1 * (1 - b_sign)) + (-1 * b_sign);

    // Add a/b values with sign bit
    split[0] = (a & 7) * a_multiplier;
    split[1] = (b & 7) * b_multiplier;

    return split;
}

// Function that takes in an array and quantizes it
int8_t* quantize_array(int8_t* arr, int8_t* quantized, int size) {
    for (int i = 0; i < size; i += 2) {
        quantized[i] = quantize(arr[i], arr[i+1]);
    }
    if (size % 2 == 1) {
        quantized[size-1] = quantize(arr[size-1], 0);
    }
    return quantized;
}


int main() {
    // list of items within quantize range
    int8_t arr[] = {0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7};
    int8_t out_arr[len(arr)];
    printf("Output Array: ");
    for (int i = 0; i < len(arr); i++) {
        printf("%d ", out_arr[i]);
    }

    delete[] split;
    delete[] arr;
    delete[] quantized;
    delete[] out_arr;
    return 0;
}

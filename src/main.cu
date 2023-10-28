#include <stdio.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_4BIT 7
#define MIN_4BIT -7

int8_t quantize(int8_t a, int8_t b, int8_t* error) {
    /*
    quantizes a pair of values to fit two 4-bit signed integers into a single byte
    */
    
    if (a > MAX_4BIT || a < MIN_4BIT || b > MAX_4BIT || b < MIN_4BIT) {
        printf("ERROR: Input integers must fit in 4 bits\n");
        *error = 1;
        return 0;
    }

    // Signed Ints: 2^3 & 2^8 ( [0]000[0]000 )
    int8_t a_val = (a >> sizeof(int)-1) & 128;
    int8_t b_val = (b >> sizeof(int)-1) & 8;

    a = abs(a);
    b = abs(b);

    // Transalte 4bit ints to left & right half of 8-bit int
    a_val += (a & 7) * 16;
    b_val += b & 7;

    return int8_t(a_val + b_val);
}

void split_ints(int8_t c, int8_t* split) {
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

    // Save 4bit int with multiplier
    split[0] = (a & 7) * a_multiplier;
    split[1] = (b & 7) * b_multiplier;
}

void quantize_array(int8_t* arr, int8_t* quantized, int8_t size, int8_t* error) {
    int8_t index = 0;
    for (int i = 0; i < size-1; i += 2) {
        quantized[index] = quantize(arr[i], arr[i+1], error);
        if (*error == 1) { return; }
        index += 1;
    }
    int8_t out_len = int(size / 2) + (size % 2);
    if (size % 2 == 1) {
        quantized[out_len-1] = quantize(arr[size-1], -0, error);
    }
}

void unquantize_array(int8_t* arr, int8_t* unquantized, int8_t size) {
    int8_t index = 0;
    for (int i = 0; i < size; i++) {
        int8_t split[2];
        split_ints(arr[i], split);
        unquantized[index] = split[0];
        unquantized[index+1] = split[1];
        index += 2;
    }
}

int main() {
    int8_t error = 0;
    // Test Quantize
    int8_t a_len = 7;
    int8_t out_len = int(a_len / 2) + (a_len % 2);

    // list of items within quantize range
    int8_t arr[a_len] = {0,-1,2,3,4,5,6};
    //  Allocate 3 bytes of memory on host for int8 arrray
    int8_t out_arr[out_len];
    quantize_array(arr, out_arr, a_len, &error);

    printf("Quantized Array: ");
    for (int i = 0; i < out_len; i++) {
        printf("%d ", out_arr[i]);
    }

    int8_t unquantized[a_len];
    unquantize_array(out_arr, unquantized, out_len);
    printf("Unquantized Array: ");
    for (int i = 0; i < a_len; i++) {
        printf("%d ", unquantized[i]);
    }
    printf("\n");

    return 0;
}

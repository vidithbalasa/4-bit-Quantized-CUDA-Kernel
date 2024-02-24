// #include "kernel.cu"
#include "../include/quantize_utils.h"

int8_t quantize_cpu(int8_t a, int8_t b) {
    /* Quantizes a pair of values to fit two 4-bit signed integers into a single byte */
    
    if (a > MAX_4BIT || a < MIN_4BIT || b > MAX_4BIT || b < MIN_4BIT) {
        printf("ERROR: Input integers must fit in 4 bits\n");
        return 0x80; // Set highest bit to indicate error
    }

    // Mask for value of signed bits: 2^3 & 2^8 ( [0]000[0]000 )
    int8_t a_val = a & 0x08;
    int8_t b_val = (b >> 4) & 0x08;

    // Masking for absolute value
    int8_t a_mask = a >> 7;
    a = (a ^ a_mask) - a_mask;
    int8_t b_mask = b >> 7;
    b = (b ^ b_mask) - b_mask;

    // Transalte 4bit ints to left & right half of 8-bit int
    a_val += (a & 7) * 16;
    b_val += b & 7;

    return int8_t(a_val + b_val);
}

void quantize_array_cpu(int8_t* arr, int8_t* quantized, int8_t size, int8_t (*quantize)(int8_t, int8_t)) {
    int8_t index = 0;
    for (int i = 0; i < size-1; i += 2) {
        quantized[index] = quantize(arr[i], arr[i+1]);
        index += 1;
    }
    int8_t out_len = int(size / 2) + (size % 2);
    if (size % 2 == 1) {
        quantized[out_len-1] = quantize(arr[size-1], -0);
    }
}

void split_ints(int8_t c, int8_t* split) {
    /* Splits a single byte into two "4-bit" signed integers */

    // Split byte
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

void unquantize_array_cpu(int8_t* arr, int8_t* unquantized, int8_t size) {
    int8_t index = 0;
    for (int i = 0; i < size; i++) {
        int8_t split[2];
        split_ints(arr[i], split);
        unquantized[index] = split[0];
        unquantized[index+1] = split[1];
        index += 2;
    }
}

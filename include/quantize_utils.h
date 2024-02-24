#ifndef QUANTIZE_UTILS_H
#define QUANTIZE_UTILS_H

#include <stdio.h>
#include <stdint.h>
#include "global.h"

int8_t quantize_cpu(int8_t a, int8_t b);
void quantize_array_cpu(int8_t* arr, int8_t* quantized, int8_t size, int8_t (*quantize)(int8_t, int8_t));
void split_ints(int8_t c, int8_t* split);
void unquantize_array_cpu(int8_t* arr, int8_t* unquantized, int8_t size);

#endif // QUANTIZE_UTILS_H

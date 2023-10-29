#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_4BIT 7
#define MIN_4BIT -7

__host__ __device__ int8_t quantize(int8_t a, int8_t b) {
    /*
    quantizes a pair of values to fit two 4-bit signed integers into a single byte
    */
    
    if (a > MAX_4BIT || a < MIN_4BIT || b > MAX_4BIT || b < MIN_4BIT) {
        printf("ERROR: Input integers must fit in 4 bits\n");
        return 0x80; // Set highest bit to indicate error
    }

    // Mask for value of signed ints: 2^3 & 2^8 ( [0]000[0]000 )
    int8_t a_val = a & 0x08;
    int8_t b_val = (b >> 4) & 0x08;

    // Masking for absolute value
    int8_t a_mask = a >> 7;
    a = (a ^ a_mask) - a_mask;
    int8_t b_mask = b >> 7;
    b = (b ^ b_mask) - b_mask;

    // transalte 4bit ints to left & right half of 8-bit int
    a_val += (a & 7) * 16;
    b_val += b & 7;

    return int8_t(a_val + b_val);
}

// __global__ void quantize_kernel(int8_t* arr, int8_t* quantized, int8_t size) {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index < size-1) {
//         quantized[index/2] = quantize(arr[index], arr[index+1]);
//     }
//     if (size % 2 == 1 && index == size-1) {
//         quantized[size/2] = quantize(arr[size-1], -0);
//     }
// }

__global__ void quantize_kernel(int8_t* arr, int8_t* quantized, int8_t size) {
    __shared__ int errorFlag;
    if (threadIdx.x == 0) {
        errorFlag = 0;
    }
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size-1) {
        int8_t q_value = quantize(arr[index], arr[index+1]);
        if (q_value & 0x80) {
            atomicAdd(&errorFlag, 1);
        } else {
            quantized[index/2] = q_value;
        }
    } else if (size % 2 == 1 && index == size-1) {
        quantized[size/2] = quantize(arr[size-1], 0);
    }

    __syncthreads();

    // Return if there's an error
    if (threadIdx.x == 0 && errorFlag) {
        printf("ERROR: Input integers must fit in 4 bits\n");
    }
}

void quantize_array(int8_t* arr, int8_t* quantized, int8_t size, int8_t* error) {
    int8_t* d_arr;
    int8_t* d_quantized;

    cudaMalloc((void**)&d_arr, size * sizeof(int8_t));
    cudaMalloc((void**)&d_quantized, ((size+1)/2) * sizeof(int8_t));

    cudaMemcpy(d_arr, arr, size * sizeof(int8_t), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    quantize_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, d_quantized, size);

    cudaMemcpy(quantized, d_quantized, ((size+1)/2) * sizeof(int8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_quantized);
}

void quantize_array_cpu(int8_t* arr, int8_t* quantized, int8_t size, int8_t* error) {
    int8_t index = 0;
    for (int i = 0; i < size-1; i += 2) {
        quantized[index] = quantize(arr[i], arr[i+1]);
        if (*error == 1) { return; }
        index += 1;
    }
    int8_t out_len = int(size / 2) + (size % 2);
    if (size % 2 == 1) {
        quantized[out_len-1] = quantize(arr[size-1], -0);
    }
}

void split_ints(int8_t c, int8_t* split) {
    /* splits a single byte into two "4-bit" signed integers */

    // split byte
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
    int a_len = 1000000;
    int8_t out_len = int(a_len / 2) + (a_len % 2);

    // create array of random ints between -7 and 7
    int8_t arr[a_len];
    for (int i = 0; i < a_len; i++) {
        arr[i] = rand() % 15 - 7;
    }
    
    //  Allocate 3 bytes of memory on host for int8 arrray
    int8_t out_arr[out_len];
    quantize_array(arr, out_arr, a_len, &error);

    // int8_t unquantized[a_len];
    // unquantize_array(out_arr, unquantized, out_len);

    // make sure there's no error
    if (error) {
        printf("Error found in main call");
    }

    // // Make sure original array matches output array
    // for (int i = 0; i < a_len; i++) {
    //     if (arr[i] != unquantized[i]) {
    //         printf("ERROR: Original array does not match unquantized array\n");
    //         return 0;
    //     }
    // }

    // printf("Quantized Array: ");
    // for (int i = 0; i < out_len; i++) {
    //     printf("%d ", out_arr[i]);
    // }
    // printf("\n");

    // printf("Unquantized Array: ");
    // for (int i = 0; i < a_len; i++) {
    //     printf("%d ", unquantized[i]);
    // }
    // printf("\n");

    return 0;
}

#include "kernel.cu"
#include "quantize_utils.cpp"
#include <iostream>

int main() {
    int a_len = 2;
    int out_len = int(a_len / 2) + (a_len % 2);

    // create array of random ints between -7 and 7
    int8_t* arr = (int8_t*)malloc(a_len * sizeof(int8_t));
    // for (int i = 0; i < a_len; i++) {
    //     arr[i] = rand() % 15 - 7;
    // }

    // Put 0 to 7 in arr
    // for (int i = 0; i < a_len; i++) {
    //     arr[i] = -i;
    // }

    // //  Allocate 3 bytes of memory on host for int8 arrray
    // int8_t out_arr[out_len];
    // quantize_array(arr, out_arr, a_len);

    // set array w 2 numbers, -3 and -4
    arr[0] = -3;
    arr[1] = -4;

    int8_t out_arr_cpu[out_len];
    quantize_array_cpu(arr, out_arr_cpu, a_len, quantize);
    printf("%d\n", out_arr_cpu[0]);

    // int8_t unquantized[a_len];
    // unquantize_array(out_arr_cpu, unquantized, a_len);

    // // Make sure original array matches output array
    // for (int i = 0; i < out_len; i++) {
    //     if (out_arr[i] != out_arr_cpu[i]) {
    //         printf("DIFFERS AT: %d != %d\n", out_arr[i], out_arr_cpu[i]);
    //     }
    //     if (out_arr[i] == 0x80 || out_arr_cpu[i] == 0x80) {
    //         printf("Error found at %d || Original Value: %d\n", i, arr[i]);
    //     }
    // }

    // // Print quantized array
    // for (int i = 0; i < out_len; i++) {
    //     printf("%d ", out_arr[i]);
    // }

    // print quantized -3 & -4
    // int t = quantize(-3, -4);
    // printf("\n%d\n", t);


    // // Print unquantized array
    // for (int i = 0; i < a_len; i++) {
    //     printf("%d ", unquantized[i]);
    // }
    // printf("\n");

    free(arr);
    return 0;
}

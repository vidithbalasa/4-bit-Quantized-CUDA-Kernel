#include "kernel.cu"
#include "quantize_utils.cpp"

int main() {
    int a_len = 10;
    int8_t out_len = int(a_len / 2) + (a_len % 2);

    // create array of random ints between -7 and 7
    int8_t* arr = (int8_t*)malloc(a_len * sizeof(int8_t));
    for (int i = 0; i < a_len; i++) {
        arr[i] = rand() % 15 - 7;
    }

    //  Allocate 3 bytes of memory on host for int8 arrray
    int8_t out_arr[out_len];
    quantize_array(arr, out_arr, a_len);

    // int8_t unquantized[a_len];
    // unquantize_array(out_arr, unquantized, a_len);

    int8_t out_arr_cpu[out_len];
    quantize_array_cpu(arr, out_arr_cpu, a_len, quantize);

    // Make sure original array matches output array
    for (int i = 0; i < out_len; i++) {
        if (out_arr[i] != out_arr_cpu[i]) {
            printf("DIFFERS AT: %d != %d\n", out_arr[i], out_arr_cpu[i]);
        }
        if (out_arr[i] == 0x80 || out_arr_cpu[i] == 0x80) {
            printf("Error found at %d || Original Value: %d\n", i, arr[i]);
        }
    }

    // // Print top 10
    // for (int i = 0; i < 10; i++) {
    //     printf("%d ", out_arr[i]);
    // }
    // printf("\n");

    free(arr);
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

int quantize(int a, int b = 0) {
    // RANGE(-7 ~ 7) -> (0 ~ 15)
    if (a > 7 || a < -7 || b > 7 || b < -7) {
        printf("ERROR: quantize out of range\n");
        exit(1);
    }

    a = a + 7;
    b = b + 7;
    return (a << 4) + b;
}

int main() {
    int a = 6;
    int b = -7;
    int c = quantize(a, b);
    printf("c = %d\n", c);

    return 0;
}

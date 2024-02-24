#include <bitset>
#include <iostream>
#include <quantize_utils.h>
#include <vector>

int main() {
    int8_t nums[20];
    int8_t quantized_nums[10];

    for (int i=0; i<20; i++) {
        nums[i] = (int8_t)i;
    }

    // quantize_array_cpu(nums, quantized_nums, 20, quantize_cpu);
    int8_t x = 1;
    int8_t y = 2;
    int8_t z = quantize_cpu(x, y);
    std::bitset<8> bits(z);
    std::cout << bits.to_string() << std::endl;

    /*
    for (int i=0; i<10; i++) {
        std::cout << quantized_nums[i] << std::endl;
    }
    */

    return 0;
}

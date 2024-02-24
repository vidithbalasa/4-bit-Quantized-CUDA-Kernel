#include <bitset>
#include <iostream>
#include <cmath>
#include <quantize_utils.h>
#include <vector>
#include <numeric>

int main() {
    int example_size = 6;
    int half_size = (int)std::ceil(example_size/2.0);
    std::vector<int8_t> nums(example_size);
    std::iota(nums.begin(), nums.end(), static_cast<int8_t>(0));
    int8_t quantized_nums[half_size];

    std::cout << "How it works\n\n"
            << "First we create an array of numbers that we want to shrink to 4 bits:\n\t";

    for (int i=0; i<example_size; i++) {
        std::cout << i;
        if (i != example_size-1) std::cout << ",        ";
    }

    std::cout << "\n\t";

    for (int i=0; i<example_size; i++) {
        std::bitset<8> bits(i);
        std::cout << bits.to_string();
        if (i != example_size-1) std::cout << ", ";
    }

    std::cout << "\n\nThen we can squish each pair of elements into a single 8 bit integer for storage:\n\t";

    quantize_array_cpu(nums.data(), quantized_nums, example_size, quantize_cpu);

    for (int i=0; i<half_size; i++) {
        std::bitset<8> bits(quantized_nums[i]);
        std::cout << bits.to_string();
        if (i != half_size-1) std::cout << ", ";
    }

    std::cout << "\n\nAfter performing any necessary calculations on the 4 bit integers, we can convert them back:\n\t";

    unquantize_array(quantized_nums, nums.data(), half_size);

    for (int i=0; i<example_size; i++) {
        std::cout << i;
        if (i != example_size-1) std::cout << ", ";
    }

    std::cout << std::endl;

    return 0;
}

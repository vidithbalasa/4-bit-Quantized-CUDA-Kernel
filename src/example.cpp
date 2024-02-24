#include <bitset>
#include <iostream>
#include <cmath>
#include <quantize_utils.h>
#include <vector>
#include <numeric>

void print_nums(std::vector<int8_t>& nums) {
    std::cout << "| ";
    for (int8_t num : nums) {
        std::cout << (int)num << "        ";
    }
    std::cout << std::endl << "\t";
}

void print_bits(std::vector<int8_t>& nums) {
    std::cout << "| ";
    for (int8_t num : nums) {
        std::bitset<8> bits(num);
        std::cout << bits.to_string() << " ";
    }
}

int main() {
    int example_size = 6;
    int half_size = (int)std::ceil(example_size/2.0);
    std::vector<int8_t> nums(example_size);
    std::iota(nums.begin(), nums.end(), 0);
    std::vector<int8_t> quantized_nums(half_size);

    std::cout << "How it works\n\n"
            << "First we create an array of numbers that we want to shrink to 4 bits:\n\t";

    print_nums(nums);
    print_bits(nums);

    std::cout << "\n\nThen we can squish each pair of elements into a single 8 bit integer for storage:\n\t";

    quantize_array_cpu(nums.data(), quantized_nums.data(), example_size, quantize_cpu);

    print_bits(quantized_nums);

    std::cout << "\n\nAfter performing any necessary calculations on the 4 bit integers, we can convert them back:\n\t";

    unquantize_array(quantized_nums.data(), nums.data(), half_size);

    print_nums(nums);

    std::cout << std::endl;

    return 0;
}

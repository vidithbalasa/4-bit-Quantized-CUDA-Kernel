#include <kernel.cuh>
#include <iostream>
#include <vector>
#include <numeric>
#include <bitset>

void quantize_array(int8_t* h_input, int8_t* h_output, int n) {
    int8_t* d_input;
    int8_t* d_output;

    // Allocate device memory
    cudaMalloc(&d_input, n * sizeof(int8_t));
    cudaMalloc(&d_output, (n / 2 + n % 2) * sizeof(int8_t));

    // Copy input data to device
    cudaMemcpy(d_input, h_input, n * sizeof(int8_t), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (n + 2 * threadsPerBlock - 1) / (2 * threadsPerBlock);

    // Launch kernel
    quantize_array_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, (n / 2 + n % 2) * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

void print_bits(std::vector<int8_t>& nums) {
    std::cout << "| ";
    for (int8_t num : nums) {
        std::bitset<8> bits(num);
        std::cout << bits.to_string() << " ";
    }
    std::cout << std::endl;
}

int main() {
    int size = 6;
    int half_size = std::ceil(size / 2.0);
    std::vector<int8_t> nums(size);
    std::iota(nums.begin(), nums.end(), 0);
    std::vector<int8_t> quantized_nums(half_size);

    print_bits(nums);

    quantize_array(nums.data(), quantized_nums.data(), size);

    print_bits(quantized_nums);

    return 0;
}

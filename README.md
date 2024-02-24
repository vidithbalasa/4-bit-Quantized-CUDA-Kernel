# 4-Bit-Quantized-CUDA-Kernel

## Overview

VRAM is expensive. But the rich people making AI keep building bigger models. Since we can't buy more GPUs, we make the weights smaller. Presenting the 4 bit kernel. Shrink the model, not your wallet.

## Technical Details

- **Quantization and Storage**: Utilizing a simple quantization algorithm, integers are translated to a 4-bit signed format. Pairs of quantized integers are then stored in an 8-bit container, allowing you to store 2x the number of weights in the same amount of space compared to an 8 bit weights.
- **Lazy Evaluation**: To combat the inherent inefficiency of operating on quantized data (the processing of unquantizing), the kernel employs lazy evaluation strategies. This approach defers the expansion and computation of quantized values until necessary, reducing memory usage and computational overhead.
- **Memory Management**: The implementation carefully manages memory to ensure that only one "full-size" 8-bit matrix is in memory at any time. This strategy optimizes both the memory footprint and computational efficiency of operations on the GPU.

## Overview
The CUDA programming model and the underlying C++ language do not natively support 4-bit integers. This poses a significant challenge for applications that can benefit from the memory efficiency of quantized data types, such as deep learning models and low-precision, high-speed arithmetic operations.

To overcome this limitation, we developed a method for representing signed 4-bit integers (int4) within the CUDA environment. Our approach involves quantizing integers to a 4-bit representation and then pairing them to be stored within a single 8-bit integer. This technique effectively doubles the memory efficiency by allowing the storage of two quantized integers in the space traditionally occupied by a single 8-bit integer.

### Memory Efficiency and Lazy Evaluation

A noteworthy aspect of our implementation is the consideration for memory usage and computational efficiency. While pairing 4-bit integers within an 8-bit container improves memory density, we further optimize resource utilization through lazy evaluation of arithmetic operations. This method ensures that data is only expanded to a full 8-bit representation when necessary for computation, minimizing memory footprint and data movement. As a result, operations are executed more swiftly, and the overhead of moving data between host and device memory is significantly reduced, as data transfer occurs once and only as required.

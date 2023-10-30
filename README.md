A functional signed 4-bit int CUDA kernel. C++ doesn't natively support 4bit ints, so I chose to save integers by first quantizing them to a signed 4-bit int and then saving pairs of them in a single 8bit int. This way every quantized int is a pair of 4-bit ints giving you about 2x. Some of this gained memory would be lost by the fact that we need 8 bits (or at least more than 4, but other values are uneven) to do any reasonable calculation on the values. To combad this I set up the functions to be evaluated lazily. This not only saves memory as you only need to track one full size 8bit matrix, but also speeds up the process as you only need to move the data from host to device once whenever its required.
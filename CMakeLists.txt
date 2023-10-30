cmake_minimum_required(VERSION 3.10)

# Name of the project
project(CUDA_4bit_kernel)

# Enable language support for CUDA
enable_language(CUDA)

# Set C++ standard (can adjust as necessary)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Find CUDA
find_package(CUDA REQUIRED)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add the executable and its sources using CUDA's utility
cuda_add_executable(${PROJECT_NAME} src/example.cu OPTIONS -arch=sm_61)

# Link CUDA device side libraries
target_link_libraries(${PROJECT_NAME} PRIVATE cudadevrt cudart_static)

# Link host libraries
target_link_libraries(${PROJECT_NAME} PRIVATE -ldl -lrt -lpthread)

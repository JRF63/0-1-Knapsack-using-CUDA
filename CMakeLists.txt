# >= 3.24 for "CUDA_ARCHITECTURES native"
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    0_1_knapsack_cuda
    VERSION 0.1.0
    DESCRIPTION "Space-efficient 0-1 knapsack problem solver"
    HOMEPAGE_URL "https://github.com/JRF63/0-1-Knapsack-using-CUDA"
    LANGUAGES CXX CUDA
)

enable_testing()

# Initialize CMAKE_CUDA_ARCHITECTURES when CMAKE_CUDA_COMPILER_ID is NVIDIA
cmake_policy(SET CMP0104 NEW)

if(BUILD_SHARED_LIBS)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

add_library(dp_cuda STATIC
    dp_cuda.cu
    dp_cuda.h
)
target_compile_features(
    dp_cuda
    PUBLIC cxx_std_11
)
set_target_properties(
    dp_cuda
    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON # Fixes linking for Windows
        CUDA_ARCHITECTURES native
)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE dp_cuda)

add_executable(simple_test simple_test.cpp)
target_link_libraries(simple_test PRIVATE dp_cuda)
add_test(
    NAME simple_test
    COMMAND simple_test
)
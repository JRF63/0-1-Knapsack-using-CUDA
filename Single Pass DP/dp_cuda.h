#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

typedef uint32_t weight_t;
typedef uint32_t value_t;
typedef uint32_t index_t;

#define HOST_MAX_MEM 5368709120 // 5GiB
#define NUM_THREADS 384
#define NUM_SEGMENTS 512

value_t gpu_knapsack(const weight_t capacity,
                     const weight_t* weights,
                     const value_t* values,
                     const index_t num_items,
                     char* taken_indices);
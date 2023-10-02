#include "dp_cuda.h"
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

// Error checking macro.
#define GPU_ERRCHECK(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Dynamic programming kernel.
__global__ void dynamic_prog(const value_t* __restrict__ prev_slice,
                             value_t* __restrict__ slice,
                             char* __restrict__ backtrack,
                             const weight_t weight,
                             const value_t value,
                             const index_t offset)
{
    const index_t j = blockDim.x * blockIdx.x + threadIdx.x + offset;
    
    value_t val_left = prev_slice[j];
    value_t val_diag = j < weight ? 0 : prev_slice[j - weight] + value;

    value_t ans;
    char bit;
    if (val_left >= val_diag) {
        ans = val_left;
        bit = 0;
    } else {
        ans = val_diag;
        bit = 1;
    }

    slice[j] = ans;
    backtrack[j] = (backtrack[j] << 1) ^ bit;
}

void backtrack_solution(const char* backtrack,
                        char* taken_indices,
                        weight_t capacity,
                        const weight_t* weights,
                        const index_t num_items)
{
    const weight_t capacity_plus_one = capacity + 1;
    const index_t last_shift = num_items % 8;
    const index_t last_idx = (num_items - 1)/8;

    // Process the last row
    for (index_t shift = 0; shift < last_shift; ++shift) {
        const char rest = backtrack[capacity_plus_one*last_idx + capacity] >> shift;
        if (rest == 0x00) {
            break;
        }
        
        if ((rest & 0x01) == 0x01) {
            const index_t i = 8*last_idx + (last_shift - shift - 1);
            taken_indices[2*i] = '1';
            capacity -= weights[i];
        }
    }
    
    // Process the rest
    for (index_t idx = (num_items - 1)/8 - 1; idx + 1 > 0; --idx) {
        for (index_t shift = 0; shift < 8; ++shift) {
            const char rest = backtrack[capacity_plus_one*idx + capacity] >> shift;
            if (rest == 0x00) {
                break;
            }
            
            if ((rest & 0x01) == 0x01) {
                const index_t i = 8*idx + (8 - shift - 1);
                taken_indices[2*i] = '1';
                capacity -= weights[i];
            }
        }
    }
}

template <typename D, D fn>
struct cudaDeleterWrapper {
    template <typename T>
    constexpr void operator()(T arg) const {
        GPU_ERRCHECK( fn(arg) );
    }
};

// std::remove_pointer<T>::type is used instead of T because of cudaStream_t
template <typename T, typename D, D fn>
using cuda_unique_ptr =
    std::unique_ptr<std::remove_pointer<T>::type, cudaDeleterWrapper<D, fn>>;

// Creates the backtrack matrix in the host. This will hold the bitset of the
// solution.
cuda_unique_ptr<char*, decltype(&cudaFreeHost), cudaFreeHost>
create_backtrack_matrix(const index_t num_items,
                        const weight_t capacity_plus_one) {
    // Calculate the minimum char's needed given that a char bitset can
    // hold 8 items
    const uint64_t num_chars = ((uint64_t)num_items - 1)/8 + 1;

    const uint64_t memory_size = (uint64_t)capacity_plus_one * num_chars;

    if (memory_size > HOST_MAX_MEM) {
        fprintf(stderr, "Exceeded memory limit");
        exit(1);
    }

    char* tmp;
    GPU_ERRCHECK( cudaMallocHost((void**)&tmp, memory_size) );
    return cuda_unique_ptr<char*, decltype(&cudaFreeHost), cudaFreeHost>(tmp);
}

// Creates the CUDA streams.
std::vector<cuda_unique_ptr<cudaStream_t, decltype(&cudaStreamDestroy),
                            cudaStreamDestroy>>
create_cuda_streams(const index_t num_streams) {
    std::vector<cuda_unique_ptr<cudaStream_t,
                                decltype(&cudaStreamDestroy),
                                cudaStreamDestroy>>
        streams;
    for (index_t i = 0; i < num_streams; ++i) {
        cudaStream_t tmp;
        GPU_ERRCHECK( cudaStreamCreate(&tmp) );
        streams.emplace_back(tmp);
    }
    return streams;
}

// Helper function for creating RAII CUDA device buffers.
template <typename T>
cuda_unique_ptr<T*, decltype(&cudaFree), cudaFree> create_gpu_buffer(
    size_t count) {
    T* tmp;
    GPU_ERRCHECK( cudaMalloc((void**)&tmp, sizeof(T)*count) );
    return cuda_unique_ptr<T*, decltype(&cudaFree), cudaFree>(tmp);
}

// GPU knapsack problem solver.
value_t gpu_knapsack(const weight_t capacity,
                     const weight_t* weights,
                     const value_t* values,
                     const index_t num_items,
                     char* taken_indices)
{
    constexpr uint32_t TOTAL_THREADS = NUM_SEGMENTS*NUM_THREADS;

    // Dynamic programming solution to 0-1 knapsack problem requires the
    // `capacity` of the knapsack to be indexable so we allocate one plus
    // the `capacity`
    const weight_t capacity_plus_one = capacity + 1;

    const index_t num_streams = (capacity_plus_one - 1)/TOTAL_THREADS + 1;

    // Allocate more than `capacity_plus_one` to avoid bounds checking in the
    // compute kernel
    const size_t gpu_buf_size = num_streams * TOTAL_THREADS;
    assert(gpu_buf_size >= capacity_plus_one);

    // Host memory buffer to hold the full solution.
    auto backtrack = create_backtrack_matrix(num_items, capacity_plus_one);
    
    // GPU memory buffers that each hold at least (`capacity` + 1) elements of
    // type `value_t`.
    auto dev_buffer_a = create_gpu_buffer<value_t>(gpu_buf_size);
    auto dev_buffer_b = create_gpu_buffer<value_t>(gpu_buf_size);

    // GPU memory buffer where the partial solution is copied. The algorithm
    // periodically copies the partial solution back to the host to minimize
    // the required VRAM.
    auto dev_backtrack = create_gpu_buffer<char>(gpu_buf_size);

    auto streams = create_cuda_streams(num_streams);

    value_t* prev = dev_buffer_b.get();
    value_t* curr = dev_buffer_a.get();
    value_t* switcher;

    // Initialize the zero'th pseudo-item. Note: this is not the same as the
    // first item
    {
        size_t second_stream_index = min((size_t)1, (size_t)num_streams - 1);
        GPU_ERRCHECK( cudaMemsetAsync(
            dev_backtrack.get(), 0, gpu_buf_size, streams[0].get()) );
        GPU_ERRCHECK( cudaMemsetAsync(
            prev, 0, gpu_buf_size, streams[second_stream_index].get()) );
    }
    
    // Main loop of the dynamic programming solution
    for (index_t i = 0; i < num_items; ++i) {
        weight_t weight = weights[i];
        value_t value = values[i];

        for (index_t j = 0; j < num_streams; ++j) {
            auto stream = streams[j].get();
            GPU_ERRCHECK( cudaStreamSynchronize(stream) );
            
            dynamic_prog<<<NUM_SEGMENTS, NUM_THREADS, 0, stream>>>(prev,
                                                                   curr,
                                                                   dev_backtrack.get(),
                                                                   weight,
                                                                   value,
                                                                   j*TOTAL_THREADS);

            // Copy backtrack matrix to host every 8 loops or if end is reached
            if (i % 8 == 7 || i == num_items - 1) {
                index_t idx = i/8;
                cudaMemcpyAsync(backtrack.get() + idx*capacity_plus_one + j*TOTAL_THREADS,
                                dev_backtrack.get() + j*TOTAL_THREADS,
                                min(TOTAL_THREADS, capacity_plus_one - j*TOTAL_THREADS),
                                cudaMemcpyDeviceToHost, stream);
            }
        }
        
        // Switch the two rows
        switcher = curr;
        curr = prev;
        prev = switcher;
    }

    // Wait for the backtrack matrix to be copied to the host
    GPU_ERRCHECK( cudaDeviceSynchronize() );
    
    backtrack_solution(backtrack.get(), taken_indices, capacity, weights, num_items);
    
    // Get the highest value in the knapsack
    value_t best;
    // If `num_items` is even, the last output is in A, otherwise it's in B
    value_t* ptr = (num_items % 2) ? dev_buffer_a.get() : dev_buffer_b.get();
    cudaMemcpy(&best,
               ptr + capacity,
               sizeof(value_t),
               cudaMemcpyDeviceToHost);
    
    return best;
}


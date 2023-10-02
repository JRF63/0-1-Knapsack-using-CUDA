#include "dp_cuda.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>

//============================ ERROR CHECKING MACRO ============================
#define GPU_ERRCHECK(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char* file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}


//============================ INITIALIZING KERNEL =============================
__global__ void initialize_ws(value_t* __restrict__ workspace,
                              char* __restrict__ backtrack,
                              const weight_t weight,
                              const value_t value,
                              const weight_t capacity,
                              const index_t offset)
{
    const index_t j = blockDim.x * blockIdx.x + threadIdx.x + offset;
    if (j <= capacity) {
        if (j >= weight) {
            workspace[j] = value;
            backtrack[j] = 1;
        } else {
            workspace[j] = 0;
            backtrack[j] = 0;
        }
    }
}


//================================= DP KERNEL ==================================
__global__ void dynamic_prog(const value_t* __restrict__ prev_slice,
                             value_t* __restrict__ slice,
                             char* __restrict__ backtrack,
                             const weight_t weight,
                             const value_t value,
                             const weight_t capacity,
                             const index_t offset)
{
    const index_t j = blockDim.x * blockIdx.x + threadIdx.x + offset;
    if (j <= capacity) {
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
}

void backtrack_solution(const char* backtrack,
                        char* taken_indices,
                        weight_t capacity,
                        const weight_t* weights,
                        const index_t num_items)
{
    const weight_t last = capacity + 1;
    const index_t last_shift = num_items % 8;
    const index_t last_idx = (num_items - 1)/8;
    //-----------------------------PROCESS LAST ROW-----------------------------
    for (index_t shift = 0; shift < last_shift; ++shift) {
        const char rest = backtrack[last*last_idx + capacity] >> shift;
        if (rest == 0x00) {
            break;
        }
        
        if ((rest & 0x01) == 0x01) {
            const index_t i = 8*last_idx + (last_shift - shift - 1);
            taken_indices[2*i] = '1';
            capacity -= weights[i];
        }
    }
    //-----------------------------PROCESS THE REST-----------------------------
    for (index_t idx = (num_items - 1)/8 - 1; idx + 1 > 0; --idx) {
        for (index_t shift = 0; shift < 8; ++shift) {
            const char rest = backtrack[last*idx + capacity] >> shift;
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

//============================ GPU CALLING FUNCTION ============================
value_t gpu_knapsack(const weight_t capacity,
                     const weight_t* weights,
                     const value_t* values,
                     const index_t num_items,
                     char* taken_indices)
{
    //---------------------------- HELPER VARIABLES-----------------------------
    const weight_t last = capacity + 1;
    const index_t num_streams = last/(NUM_SEGMENTS*NUM_THREADS) + 1;
    
    //------------------------------ HOST SET-UP -------------------------------
    const uint64_t memory_size = (uint64_t)last * (uint64_t)((num_items - 1)/8 + 1);
    if (memory_size > HOST_MAX_MEM) {
        fprintf(stderr, "Exceeded memory limit");
        exit(1);
    }
    
    char* backtrack;
    GPU_ERRCHECK( cudaMallocHost((void**)&backtrack, memory_size) );
 
    cudaStream_t* streams = (cudaStream_t*) malloc(sizeof(cudaStream_t) * num_streams);
    for (index_t i = 0; i < num_streams; ++i) {
        GPU_ERRCHECK( cudaStreamCreate(streams + i) );
    }

    //------------------------------- GPU SET-UP -------------------------------
    value_t* dev_workspace;
    char* dev_backtrack;
    GPU_ERRCHECK( cudaMalloc((void**)&dev_workspace, sizeof(value_t)*2*last) );
    GPU_ERRCHECK( cudaMalloc((void**)&dev_backtrack, last) );
    
    value_t* prev = dev_workspace;
    value_t* curr = dev_workspace + last;
    value_t* switcher;
    
    //-------------------------- INITIALIZE FIRST ROW --------------------------
    weight_t weight = weights[0];
    value_t value = values[0];
    for (index_t j = 0; j < num_streams; ++j) {
        initialize_ws<<<NUM_SEGMENTS, NUM_THREADS, 0, streams[j]>>>(prev,
                                                                    dev_backtrack,
                                                                    weight, value, capacity,
                                                                    j*NUM_SEGMENTS*NUM_THREADS);
    }
    
    //-------------------------MAIN LOOP OF DP KNAPSACK-------------------------
    for (index_t i = 1; i < num_items; ++i) {
        weight = weights[i];
        value = values[i];
        
        for (index_t j = 0; j < num_streams; ++j) {
            dynamic_prog<<<NUM_SEGMENTS, NUM_THREADS, 0, streams[j]>>>(prev,
                                                                       curr,
                                                                       dev_backtrack,
                                                                       weight, value, capacity,
                                                                       j*NUM_SEGMENTS*NUM_THREADS);

            //-------------COPY EVERY 8 LOOPS OR IF END IS REACHED--------------
            if (i % 8 == 7 || i == num_items - 1) {
                index_t idx = i/8;
                cudaMemcpyAsync(backtrack + idx*last + j*NUM_SEGMENTS*NUM_THREADS,
                                dev_backtrack + j*NUM_SEGMENTS*NUM_THREADS,
                                min(NUM_SEGMENTS*NUM_THREADS, last - j*NUM_SEGMENTS*NUM_THREADS),
                                cudaMemcpyDeviceToHost, streams[j]);
            }
        }
        //-------------------------SWITCH THE TWO ROWS--------------------------
        switcher = curr;
        curr = prev;
        prev = switcher;
        
        cudaDeviceSynchronize();
    }
    
    backtrack_solution(backtrack, taken_indices, capacity, weights, num_items);
    
    //------------------GET THE HIGHEST VALUE IN THE KNAPSACK-------------------
    value_t pos = (num_items % 2) ? 0 : last;
    value_t best;
    cudaMemcpy(&best,
               dev_workspace + pos + capacity,
               sizeof(value_t),
               cudaMemcpyDeviceToHost);
    
    //------------------------------FREE MEMORIES-------------------------------
    GPU_ERRCHECK( cudaFreeHost(backtrack) );
    for (index_t i = 0; i < num_streams; ++i) {
        GPU_ERRCHECK( cudaStreamDestroy(streams[i]) );
    }
    free(streams);
    GPU_ERRCHECK( cudaFree(dev_workspace) );
    GPU_ERRCHECK( cudaFree(dev_backtrack) );
    
    return best;
}


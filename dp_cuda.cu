#include "dp_cuda.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

//============================ ERROR CHECKING MACRO ============================
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line)
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
            backtrack[j] = '1';
        } else {
            workspace[j] = 0;
            backtrack[j] = '0';
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
    value_t val_left;
    value_t val_diag;
    if (j <= capacity) {
        val_left = prev_slice[j];
        val_diag = j < weight ? 0 : prev_slice[j - weight] + value;
        if (val_left >= val_diag) {
            slice[j] = val_left;
            backtrack[j] = '0';
        } else {
            slice[j] = val_diag;
            backtrack[j] = '1';
        }
    }
}

void backtrack_solution(const char* backtrack,
                        char* taken_indices,
                        weight_t capacity,
                        const weight_t* weights,
                        const index_t num_items)
{
    const weight_t last = capacity + 1;
    for (index_t i = num_items - 1; i > 0; --i) {
        if ('1' == (taken_indices[2*i] = backtrack[last*i + capacity])) {
            capacity -= weights[i];
        }
    }
    taken_indices[0] = backtrack[capacity];
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
    if (last*num_items > HOST_MAX_MEM) {
        fprintf(stderr, "Exceeded memory limit");
        exit(1);
    }
    
    char* backtrack = (char*) malloc(last*num_items);
    
    cudaStream_t* streams = (cudaStream_t*) malloc(sizeof(cudaStream_t)*num_streams);
    for (index_t i = 0; i < num_streams; ++i) {
        gpuErrchk( cudaStreamCreate(streams + i) );
    }

    //------------------------------- GPU SET-UP -------------------------------
    value_t* dev_workspace;
    char* dev_backtrack;
    gpuErrchk( cudaMalloc((void**)&dev_workspace, sizeof(value_t)*2*last) );
    gpuErrchk( cudaMalloc((void**)&dev_backtrack, last) );
    
    //-------------------------- INITIALIZE FIRST ROW --------------------------
    weight_t weight = weights[0];
    value_t value = values[0];
    for (index_t j = 0; j < num_streams; ++j) {
        initialize_ws<<<NUM_SEGMENTS, NUM_THREADS, 0, streams[j]>>>(dev_workspace,
                                                                    dev_backtrack,
                                                                    weight, value, capacity,
                                                                    j*NUM_SEGMENTS*NUM_THREADS);
        cudaMemcpyAsync(backtrack + j*NUM_SEGMENTS*NUM_THREADS,
                        dev_backtrack + j*NUM_SEGMENTS*NUM_THREADS,
                        min(NUM_SEGMENTS*NUM_THREADS, last - j*NUM_SEGMENTS*NUM_THREADS),
                        cudaMemcpyDeviceToHost, streams[j]);
    }
    
    index_t prev;
    index_t curr;
    for (index_t i = 1; i < num_items; ++i) {
        if (i % 2) {
            prev = 0;
            curr = last;
        } else {
            prev = last;
            curr = 0;
        }
        
        weight = weights[i];
        value = values[i];
        
        for (index_t j = 0; j < num_streams; ++j) {
            dynamic_prog<<<NUM_SEGMENTS, NUM_THREADS, 0, streams[j]>>>(dev_workspace + prev,
                                                                       dev_workspace + curr,
                                                                       dev_backtrack,
                                                                       weight, value, capacity,
                                                                       j*NUM_SEGMENTS*NUM_THREADS);
            cudaMemcpyAsync(backtrack + i*last + j*NUM_SEGMENTS*NUM_THREADS,
                            dev_backtrack + j*NUM_SEGMENTS*NUM_THREADS,
                            min(NUM_SEGMENTS*NUM_THREADS, last - j*NUM_SEGMENTS*NUM_THREADS),
                            cudaMemcpyDeviceToHost, streams[j]);
        }
        cudaDeviceSynchronize();
    }
    
    backtrack_solution(backtrack, taken_indices, capacity, weights, num_items);
    
    if (num_items % 2) {
        curr = 0;
    } else {
        curr = last;
    }
    value_t best;
    cudaMemcpy(&best,
               dev_workspace + curr + capacity,
               sizeof(value_t),
               cudaMemcpyDeviceToHost);
    
    //------------------------------ FREE MEMORY -------------------------------
    free(backtrack);
    for (index_t i = 0; i < num_streams; ++i) {
        gpuErrchk( cudaStreamDestroy(streams[i]) );
    }
    free(streams);
    gpuErrchk( cudaFree(dev_workspace) );
    gpuErrchk( cudaFree(dev_backtrack) );
    return best;
}


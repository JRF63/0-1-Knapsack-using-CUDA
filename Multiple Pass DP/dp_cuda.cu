#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "dp_cuda.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort)
          exit(code);
   }
}

using namespace std;

__global__ void dynamic_prog(const int* __restrict__ prev_slice, int* __restrict__ slice, int* __restrict__ last_col,
                             const int weight, const int value, const int capacity, const int offset)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x + offset;
    int new_val;
    if (j <= capacity) {
        if (j - weight < 0)
            new_val = prev_slice[j];
        else
            new_val = max(prev_slice[j], prev_slice[j - weight] + value);
        slice[j] = new_val;
        if (j == capacity)
            *last_col = new_val;
    }
}

int multi_pass_dp(const int& capacity, vector<Item>& items, vector<int>& taken_indices)
{
    #define NUM_THREADS 384
    #define NUM_SEGMENTS 512
    #define COL_BYTES (sizeof(int) * (last_index + 1))
    
    int best = 0;
    
    int curr_capacity = capacity;
    int last_index = items.size();
    
    int* last_col = (int*) malloc(COL_BYTES);

    // GPU set-up
    int* dev_workspace;
    int* dev_last_col;
    gpuErrchk( cudaMalloc((void**) &dev_workspace, sizeof(int) * 2*(curr_capacity+1)) );
    gpuErrchk( cudaMalloc((void**) &dev_last_col, COL_BYTES) );
    int* prev_slice;
    int* slice;
    int* switcher;
    
    while (last_index > 0) {
        
        prev_slice = dev_workspace;
        slice = dev_workspace + (curr_capacity + 1);
        
        cudaMemset(dev_workspace, 0, sizeof(int) * (curr_capacity+1));
        cudaMemset(dev_last_col, 0, COL_BYTES);
        memset(last_col, 0, COL_BYTES);
        
        for (int i = 1; i <= last_index; ++i) {
            const int weight = items[i-1].weight;
            const int value = items[i-1].value;
                        
            for (int offset = 0; offset < curr_capacity; offset += NUM_SEGMENTS*NUM_THREADS)
                dynamic_prog<<<NUM_SEGMENTS, NUM_THREADS>>>(prev_slice, slice, dev_last_col + i, weight, value, curr_capacity, offset);
            
            switcher = prev_slice;
            prev_slice = slice;
            slice = switcher;
        }
        
        cudaMemcpy(last_col, dev_last_col, COL_BYTES, cudaMemcpyDeviceToHost);
        
        if (curr_capacity == capacity) {
            int offset_last = (items.size() % 2) ? (2*curr_capacity + 1) : curr_capacity;
            cudaMemcpy(&best, dev_workspace + offset_last, sizeof(int), cudaMemcpyDeviceToHost);
        }
        
        int i = last_index;
        while (i > 0 && last_col[i] == last_col[i-1]) {
            --i;
        }
        
        if (i > 0)
            taken_indices.push_back(i-1);
        
        curr_capacity -= items[i-1].weight;
        last_index = i - 1;
    }
    
    free(last_col);
    
    // free GPU memory
    gpuErrchk( cudaFree(dev_workspace) );
    gpuErrchk( cudaFree(dev_last_col) );
    
    return best;
}
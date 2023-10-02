#include "dp_cuda.h"

int main() {
    constexpr int NUM_ITEMS = 4;

    index_t num_items = NUM_ITEMS;
    weight_t capacity = 11;
    value_t values[] = {8, 10, 15, 4};
    weight_t weights[] = {4, 5, 8, 3};
    char taken_indices[2 * NUM_ITEMS] = "0 0 0 0";

    value_t best =
        gpu_knapsack(capacity, weights, values, num_items, taken_indices);

    value_t ans_best = 19;
    char ans_taken_indices[2 * NUM_ITEMS] = "0 0 1 1";

    if (best != ans_best) {
        return 1;
    }
    for (int i = 0; i < 2 * NUM_ITEMS; i++) {
        if (taken_indices[i] != ans_taken_indices[i]) {
            return 2;
        }
    }

    return 0;
}
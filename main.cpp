#include "dp_cuda.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>

using namespace std;

void print_solution(const value_t best, char* taken_indices) {
    printf("%d %d\n", best, 1);
    printf("%s\n", taken_indices);
}

void parse_input(const char* filename, index_t& num_items, weight_t& capacity,
                 unique_ptr<weight_t[]>& weights,
                 unique_ptr<value_t[]>& values) {
    ifstream input_file(filename);

    if (!(input_file >> num_items >> capacity)) {
        fprintf(stderr, "Bad file format.\n");
        exit(1);
    }

    weights = make_unique<weight_t[]>(num_items);
    values = make_unique<value_t[]>(num_items);

    weight_t weight;
    value_t value;

    for (index_t i = 0; i < num_items; ++i) {
        input_file >> value >> weight;
        weights[i] = weight;
        values[i] = value;
    }
}

int main(int argc, char* argv[]) {
    value_t best;
    index_t num_items;
    weight_t capacity;
    unique_ptr<weight_t[]> weights;
    unique_ptr<value_t[]> values;
    parse_input(argv[1], num_items, capacity, weights, values);

    unique_ptr<char[]> taken_indices = make_unique<char[]>(2 * num_items);
    memset(taken_indices.get(), ' ', 2 * num_items);
    taken_indices[2 * num_items - 1] = '\0';
    for (index_t i = 0; i < num_items; ++i) {
        taken_indices[2 * i] = '0';
    }

    best = gpu_knapsack(
        capacity, weights.get(), values.get(), num_items, taken_indices.get());
    print_solution(best, taken_indices.get());

    return 0;
}
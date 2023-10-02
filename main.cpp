#include "dp_cuda.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>

void print_usage() {
    fprintf(stderr, "Usage: main.exe [INPUT_FILE]\n");
}

void print_solution(const value_t best, char* taken_indices) {
    printf("%d %d\n", best, 1);
    printf("%s\n", taken_indices);
}

void parse_input(const char* filename, index_t& num_items, weight_t& capacity,
                 std::unique_ptr<weight_t[]>& weights,
                 std::unique_ptr<value_t[]>& values) {
    std::ifstream input_file;
    input_file.exceptions(std::ifstream::badbit | std::ifstream::failbit);
    input_file.open(filename);

    // Read the header
    input_file >> num_items >> capacity;

    weights = std::make_unique<weight_t[]>(num_items);
    values = std::make_unique<value_t[]>(num_items);

    weight_t weight;
    value_t value;

    // Read the items
    for (index_t i = 0; i < num_items; ++i) {
        input_file >> value >> weight;
        weights[i] = weight;
        values[i] = value;
    }
}

std::unique_ptr<char[]> create_solution_string(index_t num_items) {
    std::unique_ptr<char[]> taken_indices = std::make_unique<char[]>(2 * num_items);

    // Set to a NULL terminated C-string of '0's with spaces in between 
    memset(taken_indices.get(), ' ', 2 * num_items);
    for (index_t i = 0; i < num_items; ++i) {
        taken_indices[2 * i] = '0';
    }
    taken_indices[2 * num_items - 1] = '\0';
    
    return taken_indices;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
    }

    index_t num_items;
    weight_t capacity;
    std::unique_ptr<weight_t[]> weights;
    std::unique_ptr<value_t[]> values;

    try {
        parse_input(argv[1], num_items, capacity, weights, values);

        auto taken_indices = create_solution_string(num_items);

        value_t best = gpu_knapsack(
            capacity, weights.get(), values.get(), num_items, taken_indices.get());
        print_solution(best, taken_indices.get());

        return 0;
    } catch (const std::system_error& e) {
        fprintf(stderr, e.what());
        return e.code().value();
    }
}
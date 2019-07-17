#include "dp_cuda.h"
#include <vector>
#include <tuple>
#include <string>
#include <iostream>
#include <fstream>

auto parse(const char* filename)
{
    std::ifstream input(filename);
    if (!input.good()) {
        std::cout << "File doesn't exist." << std::endl;
        exit(1);
    }
    
    index_t num_items;
    weight_t capacity;
    input >> num_items >> capacity;
    
    std::vector<weight_t> weights;
    std::vector<value_t> values;
    
    for (index_t i = 0; i < num_items; ++i) {
        weight_t weight;
        value_t value;
        input >> value >> weight;
        weights.push_back(weight);
        values.push_back(value);
    }
    
    return std::make_tuple(num_items, capacity, weights, values);
}

void print_solution(const value_t best, std::string& taken_indices)
{
    std::cout << best << " 1" << std::endl;
    std::cout << taken_indices << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " filename" << std::endl;
        exit(1);
    }
    
    index_t num_items;
    weight_t capacity;
    std::vector<weight_t> weights;
    std::vector<value_t> values;
    std::tie(num_items, capacity, weights, values) = parse(argv[1]);
    
    std::string taken_indices(2*num_items - 1, ' ');
    for (index_t i = 0; i < num_items; ++i) {
        taken_indices[2*i] = '0';
    }

    value_t best = gpu_knapsack(capacity,
                                weights.data(), values.data(), num_items,
                                const_cast<char*>(taken_indices.data()));
    print_solution(best, taken_indices);

    return 0;
}
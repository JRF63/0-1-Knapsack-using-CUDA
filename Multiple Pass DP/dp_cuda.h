#pragma once
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

struct Item {
    int value;
    int weight;
    float ratio;
};

int multi_pass_dp(const int& capacity, std::vector<Item>& items, std::vector<int>& taken_indices);
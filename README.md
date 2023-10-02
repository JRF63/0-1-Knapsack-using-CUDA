# 0-1 knapsack problem solver

A dynamic programming solver for the 0-1 knapsack problem written in CUDA.

This solver saves VRAM in two ways:
- Only keeping track of the previous and current solution slices (the minimum) instead of the whole CAPACITY * NUM_ITEMS matrix
- Using a bitset for the backtracking matrix

## Building

Create a directory `build` and run CMake there.
```
mkdir build
cd build
cmake ..
```
Then run the generated build system.

## Input/Output

The solver was originally made for the University of Melbourne's Discrete Optimization course in Coursera back in 2014. Hence, the input and output follows the requirement in that course.

The input is a file with a header of (`num_items`, `knapsack_capacity`) followed by `num_items` pairs of (`value`, `weight`). An example file for a knapsack capacity of 11 with 4 items:
```
4 11
8 4
10 5
15 8
4 3
```

This program output starts with a header of (`value_of_best_solution`, `is_optimal`). The solver calculates the actual best solution and not just a heuristically derived one so `is_optimal` is always 1. The header is followed by a bit vector of the items to be put in the knapsack. The solution to the example input is as follows:
```
19 1
0 0 1 1
```
By inspection we can see that the optimal solution is to take the 3rd and 4th item for a total value of 19 and total weight of exactly 11.
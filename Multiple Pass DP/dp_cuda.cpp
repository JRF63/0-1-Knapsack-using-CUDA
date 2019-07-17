#include <iostream>
#include <fstream>
#include <tuple>
#include "dp_cuda.h"

using namespace std;

void print_solution(const int best, vector<int>& taken_indices,
                    const int& num_items, bool optimal = true)
{
    cout << best << " " << (optimal ? 1 : 0) << endl;
    
    vector<int> solution(num_items, 0);
    for (const auto& index: taken_indices)
        solution[index] = 1;
    
    for (auto i = solution.begin(); i != solution.end(); ++i) {
        if (i == solution.end() - 1)
            cout << *i << endl;
        else
            cout << *i << " ";
    }
}

tuple<int, vector<Item>> parse_input(char *filename)
{
    ifstream input_file(filename);
    
    int num_items;
    int capacity;
    
    if (!(input_file >> num_items >> capacity))
        exit(1);
    
    int value;
    int weight;
    
    vector<Item> items;
    while (input_file >> value >> weight)
        items.push_back( {value, weight, static_cast<float>(value)/weight} );
    
    if (num_items != items.size()) {
        cerr << "Bad file. Expected " << num_items;
        cerr << " items but got " << items.size();
        cerr << "." << endl;
        exit(1);
    }
    
    return make_tuple(capacity, items);
}

int main(int argc, char *argv[])
{
    int capacity;
    vector<Item> items;
    tie(capacity, items) = parse_input(argv[1]);
    
    int best;
    vector<int> taken_indices;
    best = multi_pass_dp(capacity, items, taken_indices);
    
    print_solution(best, taken_indices, items.size());
    
    return 0;
}
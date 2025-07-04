#include <iostream>
#include <vector>
#include "linalg/tensor.h"
#include <algorithm> 
#include <unordered_map>
#include <string>
#include "linalg/transformer.h"

int main(){

    /*
    
    Tensor layer1(1, 5);
    layer1.randInit();
    Tensor layer2(5, 10);
    layer2.randInit();
    Tensor v3 = layer1.matMul(layer2);
    v3.randInit();
    std::cout << layer1;
    std::cout << "*" << std::endl;
    std::cout << layer2;
    std::cout << "=" << std::endl;

    std::cout << v3;
    std::cout << "ReLU" << std::endl;
    v3.ReLU();
    std::cout << v3 << std::endl;
    //v3.loadWeights("./test.txt");
    std::cout << "+" << std::endl;
    Tensor v4(1, 10);
    v4.randInit();
    std::cout << v4 << std::endl;
    Tensor v5 = v4 + v3;
    std::cout << "=" << std::endl;
    std::cout << v5 << std::endl;
    std::cout << "Softmax" << std::endl;
    v5.Softmax();
    std::cout << v5 << std::endl;
    v5.loadWeights("./weights/net_1_weight.csv");
    --------------------------------------------------------------
 
    Tensor v1("./weights/test_data.csv", 10000, 784);
    std::cout << "Loaded data" << std::endl;
    Tensor w1("./weights/net_1_weight.csv", 784, 128);
    Tensor b1("./weights/net_1_bias.csv", 1, 128);
    Tensor w2("./weights/net_3_weight.csv", 128, 64);
    Tensor b2("./weights/net_3_bias.csv", 1, 64);
    Tensor w3("./weights/net_5_weight.csv", 64, 10);
    Tensor b3("./weights/net_5_bias.csv", 1, 10);
    Tensor layer1 = v1.matMul(w1);
    layer1 = layer1 + b1;
    layer1.ReLU();
    Tensor layer2 = layer1.matMul(w2);
    layer2 = layer2 + b2;
    layer2.ReLU();
    Tensor layer3 = layer2.matMul(w3);
    layer3 = layer3 + b3;
    layer3.Softmax();
    std::cout << "Output: " << layer3 << std::endl;
    for (int i = 0; i < ((*layer3.vals).size()); i++) {
        auto max_it = std::max_element((*layer3.vals)[i].begin(), (*layer3.vals)[i].end());
        if (max_it != (*layer3.vals)[i].end()) {
            int max_index = std::distance((*layer3.vals)[i].begin(), max_it);
            std::cout << "Max value: " << *max_it << ", at index: " << max_index << std::endl;
        }
    }
    */
    std::unordered_map<std::string, int> config;
    config["n_emb_h"] = 5;
    Tensor v1(5, 2);
    v1.randInit();
    std::cout << "Tensor v1:" << std::endl;
    std::cout << v1 << std::endl;
    std::cout << "Transposing v1:" << std::endl;
    v1.transpose();
    std::cout << v1 << std::endl;
    CausalSelfAttentionSingleHead csa(config);
    Tensor result = csa.forward(v1);
    std::cout << "Result:" << std::endl;
    std::cout << result << std::endl;
}
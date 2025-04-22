#include <iostream>
#include "linalg/tensor.h"

int main(){
    std::vector<std::vector<float> > v1;
    fillMatrix(v1, 10, 5);
    Tensor layer1(v1, 10, 5);
    std::vector<std::vector<float> > v2;
    fillMatrix(v2, 5, 10);
    Tensor layer2(v2, 5, 10);
    Tensor v3 = layer1.matMul(layer2);
    std::cout << layer1;
    std::cout << "*" << std::endl;
    std::cout << layer2;
    std::cout << "=" << std::endl;
    std::cout << v3;
}
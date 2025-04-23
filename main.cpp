#include <iostream>
#include "linalg/tensor.h"

int main(){
    Tensor t1 = Tensor(50, 512);
    Tensor t2 = Tensor(50, 512);
    std::string path = "./model_info/weights/l1.weight.csv";
    loadWeights(t1, path);
    t2.fillTensor();
    std::cout << t1;
    std::cout << std::endl;
    t2.transpose();
    std::cout << t2;
    std::cout << "=" << std::endl;
    Tensor t3 = t1.matMul(t2);
    std::cout << t3;
    std::cout << t3.shape() << std::endl;
}
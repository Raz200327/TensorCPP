#include <iostream>
#include "linalg/tensor.h"

int main(){
    Tensor t1 = Tensor(2, 5);
    std::cout << t1;
    t1.transpose();
    std::cout << t1;
    
}
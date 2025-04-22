#include <iostream>
#include <vector>
#include <random>
#include "linalg/tensor.h"

std::ostream& operator<<(std::ostream& os, const Tensor& obj){
    os << "[";
    for (int i = 0; i < obj.h; i++){
        for (int a = 0; a < obj.w; a++){
            if (i > 0 && a == 0){
                os << " ";
            }
            if ((a == obj.w - 1) && (i == obj.h - 1)){
                os << obj.vals[i][a] << "]\n";
            }
            else if (a == obj.w - 1){
                os << obj.vals[i][a] << ",\n";
            }
            else {
                os << obj.vals[i][a] << ", ";
            }
        }
    }
    return os;
}


Tensor Tensor::matMul(const Tensor &v2){
    std::vector<std::vector<float> > result(this->vals.size(), std::vector<float>(v2.vals[0].size()));
    if (v2.vals.size() != this->vals[0].size()){
        std::cerr << "Error cannot multiply matrices with sizes: " << "(" << this->vals.size() << ", " << this->vals[0].size() <<
        ") " << "(" << v2.vals.size() << ", " << v2.vals[0].size() << ")" << std::endl;
        std::exit(1);
    }
    for (int i = 0; i < this->vals.size(); i++){
        for (int a = 0; a < v2.vals[0].size(); a++){
            float val = 0;
            for (int j = 0; j < this->vals[0].size(); j++){
                val += this->vals[i][j] * v2.vals[j][a];
            }
            result[i][a] = val;
        }
    }
    Tensor newTensor(result, this->vals.size(), v2.vals[0].size());
    return newTensor;
}  


void fillMatrix(std::vector<std::vector<float> >& mat, int h, int w){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(1, 10);
    mat.resize(h, std::vector<float>(w));
    for (int i = 0; i < h; i++){
        for (int a = 0; a < w; a++){
            mat[i][a] = dist(gen);
        }
    }

}


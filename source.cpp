#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include "linalg/tensor.h"

std::ostream& operator<<(std::ostream& os, const Tensor& obj){
    os << "[[";
    for (int i = 0; i < obj.h; i++){
        for (int a = 0; a < obj.w; a++){
            if (i > 0 && a == 0){
                os << "[";
            }
            if ((a == obj.w - 1) && (i == obj.h - 1)){
                os << (*obj.vals)[i][a] << "]]";
            }
            else if (a == obj.w - 1){
                os << (*obj.vals)[i][a] << "]";
            }
            else {
                os << (*obj.vals)[i][a] << ", ";
            }
        }
        os << std::endl;
    }
    return os;
}


Tensor::Tensor(std::vector<std::vector<float> > *vals, int h, int w){
    this->vals = vals;
    this->h = h;
    this->w = w;
}

Tensor Tensor::matMul(const Tensor &v2){
    std::vector<std::vector<float> > *result = new std::vector<std::vector<float> >(this->vals->size(), std::vector<float>((*v2.vals)[0].size()));
    if ((*v2.vals).size() != (*this->vals)[0].size()){
        std::cerr << "Error cannot multiply matrices with sizes: " << "(" << (*this->vals).size() << ", " << (*this->vals)[0].size() <<
        ") " << "(" << (*v2.vals).size() << ", " << (*v2.vals)[0].size() << ")" << std::endl;
        std::exit(1);
    }
    for (int i = 0; i < this->vals->size(); i++){
        for (int a = 0; a < (*v2.vals)[0].size(); a++){
            float val = 0;
            for (int j = 0; j < (*this->vals)[0].size(); j++){
                val += (*this->vals)[i][j] * (*v2.vals)[j][a];
            }
            (*result)[i][a] = val;
        }
    }
    Tensor newTensor(result, this->vals->size(), (*v2.vals)[0].size());
    return newTensor;
}  

Tensor::Tensor(int h, int w){
    this->vals = new std::vector<std::vector<float> >(h, std::vector<float>(w, 0.0f));
    this->h = h;
    this->w = w;
}

Tensor::~Tensor(){
    delete this->vals;
}

void Tensor::transpose(){
    
    std::vector<std::vector<float> >* temp = new std::vector<std::vector<float> >(this->w, std::vector<float>(this->h, 0.0f));
    for (int i = 0; i < this->h; i++){
        for (int a = 0; a < this->w; a++){
            (*temp)[a][i] = (*this->vals)[i][a];
        }
    }
    delete this->vals;
    this->vals = temp;
    int w_temp = this->w;
    this->w = this->h;
    this->h = w_temp;
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

void loadWeights(Tensor &tensor, std::string file_path){
    std::ifstream file(file_path);
    std::string line;
}


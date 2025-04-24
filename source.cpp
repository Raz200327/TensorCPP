#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <sstream>
#include <tuple>
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

void ReLU(Tensor &tensor){
    for (int i = 0; i < tensor.h; i++){
        for (int a = 0; a < tensor.w; a++){
            if ((*tensor.vals)[i][a] < 0){
                (*tensor.vals)[i][a] = 0;
            }
        }
    }
}

Tensor Tensor::matMul(const Tensor &v2){
    std::vector<std::vector<float> > *result = new std::vector<std::vector<float> >(this->vals->size(), std::vector<float>((*v2.vals)[0].size()));
    if ((*v2.vals).size() != (*this->vals)[0].size()){
        std::cerr << "Error cannot multiply matrices with sizes: " << this->shape() << " and " << v2.shape() << std::endl;
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

void Tensor::fillTensor(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < this->h; i++){
        for (int a = 0; a < this->w; a++){
            (*this->vals)[i][a] = dist(gen);
        }
    }
}

std::string Tensor::shape() const {
    return "(" + std::to_string(this->h) + ", " + std::to_string(this->w) + ")";
}

void loadWeights(Tensor &tensor, std::string file_path){

    std::ifstream file(file_path);
    std::string line;
    if (!file.is_open()) {
        std::cerr << "Could not open the file " << file_path << std::endl;
        
    }
    int i = 0;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        int j = 0;
        while (std::getline(ss, value, ',')) {
            (*tensor.vals)[i][j] = std::stof(value);
            j++;
        }
        i++;
    }

    file.close();
}


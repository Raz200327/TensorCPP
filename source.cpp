#include <iostream>
#include <vector>
#include <random>
#include <sstream>
#include <fstream>
#include "linalg/tensor.h"


Tensor::Tensor(int h, int w){
    if ((w == 0) || (h == 0)){
        std::cerr << "Cannot create tensor with dimension " << h << ", " << w << std::endl;
        std::exit(1);
    }
    this->vals = std::make_shared<std::vector<std::vector<float> > >(h, std::vector<float>(w, 0.0f));
    this->w = w;
    this->h = h;
}

Tensor::Tensor(std::shared_ptr<std::vector<std::vector<float> > > vals, int h, int w){
    if ((w == 0) || (h == 0)){
        std::cerr << "Cannot create tensor with dimension " << h << ", " << w << std::endl;
        std::exit(1);
    }
    this->vals = vals;
    this->h = h;
    this->w = w;
}

Tensor::Tensor(std::string path, int h, int w){
    this->w = w;
    this->h = h;
    std::ifstream file(path);
    if (!file.is_open()){
        std::cerr << "Unable to open file!" << std::endl;
        std::exit(1);
    }
    std::string line;
    this->vals = std::make_shared<std::vector<std::vector<float> > >(h, std::vector<float>(w, 0.0f));
    int h_i = 0;
    int w_i = 0;
    while (std::getline(file, line)) {

        std::vector<float> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            (*this->vals)[h_i][w_i] = std::stof(cell);
            w_i++;
        }
        w_i = 0;
        h_i++;
    }              
}

Tensor::~Tensor(){
    std::cout << "Tensor with size (" << this->h << ", " << this->w << ") is being deleted" << std::endl;
}

std::ostream& operator<<(std::ostream& os, const Tensor& obj){
    os << "[[";
    for (int i = 0; i < obj.h; i++){
        for (int a = 0; a < obj.w; a++){
            if (i > 0 && a == 0){
                os << " ";
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


Tensor Tensor::matMul(const Tensor &v2){
    
    std::shared_ptr<std::vector<std::vector<float> > > result = std::make_shared<std::vector<std::vector<float> > >
    (this->vals->size(), std::vector<float>(v2.vals->at(0).size()));
    if (v2.vals->size() != this->vals->at(0).size()){
        std::cerr << "Error cannot multiply matrices with sizes: " << "(" << this->vals->size() << ", " << this->vals->at(0).size() <<
        ") " << "(" << v2.vals->size() << ", " << v2.vals->at(0).size() << ")" << std::endl;
        std::exit(1);
    }
    for (int i = 0; i < this->vals->size(); i++){
        for (int a = 0; a < v2.vals->at(0).size(); a++){
            float val = 0;
            for (int j = 0; j< this->vals->at(0).size(); j++){
                val += (*this->vals)[i][j] * (*v2.vals)[j][a]; 
            }
            (*result)[i][a] = val;
        }
    }
    Tensor newTensor(result, this->vals->size(), v2.vals->at(0).size());
    return newTensor;
}  


void Tensor::randInit(){
    
    int h = this->h;
    int w = this->w;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(-10,10);
    for (int i = 0; i < h; i++){
        for (int a = 0; a < w; a++){
            (*this->vals)[i][a] = dist(gen);
        }
    }

}



Tensor operator+(const Tensor& a, const Tensor& b){
    if (!(a.w == b.w)){
        std::cerr << "Incorrect shape!" << std::endl;
        std::exit(1);
    } else if ((a.h != b.h) && (b.h == 1)){
        Tensor result(a.h, a.w);
        for (int h = 0; h < a.h; h++){
            for (int w = 0; w < a.w; w++){
                (*result.vals)[h][w] = (*a.vals)[h][w] + (*b.vals)[0][w];
            }
        }
        return result;
    } else {
        Tensor result(a.h, a.w);
        for (int h = 0; h < a.h; h++){
            for (int w = 0; w < a.w; w++){
                (*result.vals)[h][w] = (*a.vals)[h][w] + (*b.vals)[h][w];
            }
        }
        return result;
    }
    
    
}


Tensor& Tensor::operator=(const Tensor& result){
    this->vals = result.vals;
    this->h = result.h;
    this->w = result.w;
    return (*this);
}

void Tensor::ReLU(){
    for (int h = 0; h < this->h; h++){
        for(int w = 0; w < this->w; w++){
            if ((*this->vals)[h][w] < 0){
                (*this->vals)[h][w] = 0;
            }
        }
    }
}

void Tensor::Softmax(){
    for (int h = 0; h < this->h; h++){
        float denomitatorSum = 0;
        for (int w2 = 0; w2 < this->w; w2++){
            denomitatorSum += std::exp((*this->vals)[h][w2]);
        }
        for (int w = 0; w < this->w; w++){
            (*this->vals)[h][w] = std::exp((*this->vals)[h][w])/denomitatorSum;
        } 
    }
}
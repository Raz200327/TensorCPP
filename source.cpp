#include <iostream>
#include <vector>
#include <random>
#include <sstream>
#include <fstream>
#include "linalg/tensor.h"
#include <limits>

//Constructors

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

Tensor::Tensor(const Tensor& other){
    this->vals = other.vals;
    this->h = other.h;
    this->w = other.w;
}


Tensor Tensor::createMask(int h, int w){
    if (h != w){
        std::cerr << "Cannot create mask with dimensions: " << h << ", " << w << std::endl;
        std::exit(1);
    }
    std::shared_ptr<std::vector<std::vector<float> > > vals = std::make_shared<std::vector<std::vector<float> > >(h, std::vector<float>(w, -std::numeric_limits<float>::infinity()));
    for (int i = 0; i < h; i++){
        for (int a = 0; a < w; a++){
            if (a <= i){
                (*vals)[i][a] = 1;
            }
        }
    }
    Tensor mask(vals, h, w);
    return mask;
}

//Operators

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


Tensor Tensor::matMul(const Tensor &v2) const{
    
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
    float scale = std::sqrt(2.0f / (h + w));
    std::normal_distribution<float> dist(0.0f, scale);
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

Tensor operator+(const Tensor& a, const float& b){
    Tensor result(a.h, a.w);
    for (int h = 0; h < a.h; h++){
        for (int w = 0; w < a.w; w++){
            (*result.vals)[h][w] = (*a.vals)[h][w] + b;
        }
    }
    return result;
}


Tensor operator-(const Tensor& a, const Tensor& b){
    if (a.h != b.h || a.w != b.w){
        std::cerr << "Incorrect shape!" << std::endl;
        std::exit(1);
    }
    Tensor result(a.h, a.w);
    for (int h = 0; h < a.h; h++){
        for (int w = 0; w < a.w; w++){
            (*result.vals)[h][w] = (*a.vals)[h][w] - (*b.vals)[h][w];
        }
    }
    return result;
}


Tensor operator-(const Tensor& a, const float& b){
    Tensor result(a.h, a.w);
    for (int h = 0; h < a.h; h++){
        for (int w = 0; w < a.w; w++){
            (*result.vals)[h][w] = (*a.vals)[h][w] - b;
        }
    }
    return result;
}


Tensor operator/(const Tensor& a, const Tensor& b){
    if (a.h != b.h || a.w != b.w){
        std::cerr << "Incorrect shape!" << std::endl;
        std::exit(1);
    }
    Tensor result(a.h, a.w);
    for (int h = 0; h < a.h; h++){
        for (int w = 0; w < a.w; w++){
            if ((*b.vals)[h][w] == 0){
                std::cerr << "Cannot divide by zero!" << std::endl;
                std::exit(1);
            }
            (*result.vals)[h][w] = (*a.vals)[h][w] / (*b.vals)[h][w];
        }
    }
    return result;
}

Tensor operator/(const Tensor& a, const float& b){
    if (b == 0){
        std::cerr << "Cannot divide by zero!" << std::endl;
        std::exit(1);
    }
    Tensor result(a.h, a.w);
    for (int h = 0; h < a.h; h++){
        for (int w = 0; w < a.w; w++){
            (*result.vals)[h][w] = (*a.vals)[h][w] / b;
        }
    }
    return result;
}

Tensor operator*(const Tensor& a, const Tensor& b){
    if (a.h != b.h || a.w != b.w){
        std::cerr << "Incorrect shape!" << std::endl;
        std::exit(1);
    }
    Tensor result(a.h, a.w);
    for (int h = 0; h < a.h; h++){
        for (int w = 0; w < a.w; w++){
            (*result.vals)[h][w] = (*a.vals)[h][w] * (*b.vals)[h][w];
        }
    }
    return result;
}

Tensor operator*(const Tensor& a, const float& b){
    Tensor result(a.h, a.w);
    for (int h = 0; h < a.h; h++){
        for (int w = 0; w < a.w; w++){
            (*result.vals)[h][w] = (*a.vals)[h][w] * b;
        }
    }
    return result;
}

Tensor& Tensor::operator=(const Tensor& result){
    this->vals = result.vals;
    this->h = result.h;
    this->w = result.w;
    return (*this);
}

Tensor& Tensor::operator=(Tensor&& result) noexcept{
    if (this != &result){
        this->vals = result.vals;
        this->h = result.h;
        this->w = result.w;
    }
    return (*this);
}

void Tensor::transpose(){
    std::shared_ptr<std::vector<std::vector<float> > > newVals = std::make_shared<std::vector<std::vector<float> > >(this->w, std::vector<float>(this->h, 0.0f));
    for (int i = 0; i < this->h; i++){
        for (int a = 0; a < this->w; a++){
            (*newVals)[a][i] = (*this->vals)[i][a];
        }
    }
    this->vals = newVals;
    std::swap(this->h, this->w);
}

//Neural Network

NeuralNetwork::NeuralNetwork(){
    for (int i = 0; i < 3; i++){
        Tensor layer(2, 2);
        layer.randInit();
        this->layers.push_back(layer);
    }
    for (int i = 0; i < 2; i++){
        this->activationFunctions.push_back(std::make_unique<ReLU>());
    }
    this->activationFunctions.push_back(std::make_unique<Softmax>());
}

Tensor NeuralNetwork::forward(const Tensor &input){
    if (this->layers.size() == 0){
        std::cerr << "No layers in the neural network!" << std::endl;
        std::exit(1);
    }
    Tensor result = input;
    for (int i = 0; i < this->layers.size(); i++){
        result = result.matMul(this->layers[i]);
        this->activationFunctions[i]->apply(result);
    }
    return result;
}

//Causal Self Attention

CausalSelfAttentionSingleHead::CausalSelfAttentionSingleHead(const std::unordered_map<std::string, int> &config)
    : q_h(Tensor(config.at("n_emb_h"), config.at("n_emb_h"))),
      k_h(Tensor(config.at("n_emb_h"), config.at("n_emb_h"))),
      v_h(Tensor(config.at("n_emb_h"), config.at("n_emb_h"))),
      config(config) {
    this->q_h.randInit();
    this->k_h.randInit();
    this->v_h.randInit();
}

Tensor CausalSelfAttentionSingleHead::forward(const Tensor &input){

    Tensor q = input.matMul(this->q_h);
    Tensor k = input.matMul(this->k_h);
    Tensor v = input.matMul(this->v_h);
    Tensor mask = Tensor::createMask(input.h, input.h);
    std::cout << "Mask:" << std::endl;
    std::cout << mask << std::endl;
    k.transpose();
    Tensor att = q.matMul(k);
    att = att / std::sqrt(this->config.at("n_emb_h"));
    att = att + mask;
    std::cout << "Att:" << std::endl;
    std::cout << att << std::endl;
    Softmax activation;
    activation.apply(att);
    std::cout << "Att after softmax:" << std::endl;
    std::cout << att << std::endl;
    Tensor result = att.matMul(v);
    return result;
}
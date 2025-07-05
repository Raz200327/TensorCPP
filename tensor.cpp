#include "linalg/tensor.h"
#include <iostream>
#include <vector>
#include <random>
#include <sstream>
#include <fstream>
#include <limits>

// Element access
float& Tensor::operator()(int i, int j){
    return (*this->vals)[i * this->w + j];
}

const float& Tensor::operator()(int i, int j) const{
    return (*this->vals)[i * this->w + j];
}

// Constructors
Tensor::Tensor(int h, int w) : h(h), w(w) {
    this->vals = std::make_shared<std::vector<float>>(h * w, 0.0f);  // Flat storage!
}

Tensor::Tensor(std::shared_ptr<std::vector<float>> vals, int h, int w) : vals(vals), h(h), w(w) {
    if ((w == 0) || (h == 0) || vals->size() != h * w){
        std::cerr << "Cannot create tensor with dimension " << h << ", " << w << std::endl;
        std::exit(1);
    }
}

Tensor::Tensor(std::string path, int h, int w) : h(h), w(w) {
    this->vals = std::make_shared<std::vector<float>>(h * w, 0.0f);
    
    std::ifstream file(path);
    if (!file.is_open()){
        std::cerr << "Unable to open file!" << std::endl;
        std::exit(1);
    }
    
    std::string line;
    int h_i = 0;
    while (std::getline(file, line) && h_i < h) {
        std::stringstream ss(line);
        std::string cell;
        int w_i = 0;
        
        while (std::getline(ss, cell, ',') && w_i < w) {
            (*this->vals)[h_i * w + w_i] = std::stof(cell);
            w_i++;
        }
        h_i++;
    }
}

Tensor::Tensor(const Tensor& other) : vals(other.vals), h(other.h), w(other.w) {}


void Tensor::concat(const Tensor& other){
    if (this->h != other.h){
        std::cerr << "Cannot concatenate tensors with different heights!" << std::endl;
        std::exit(1);
    }
    Tensor result(this->h, this->w + other.w);
    for (int i = 0; i < this->h; i++){
        for (int j = 0; j < this->w; j++){
            (*result.vals)[i * (this->w + other.w) + j] = (*this->vals)[i * this->w + j];
        }
    }
    for (int i = 0; i < this->h; i++){
        for (int j = 0; j < other.w; j++){
            (*result.vals)[i * (this->w + other.w) + this->w + j] = (*other.vals)[i * other.w + j];
        }
    }
    this->vals = result.vals;
    this->w = result.w;
}

Tensor Tensor::matMul(const Tensor& other) const {
    if (this->w != other.h) {
        std::cerr << "Matrix dimension mismatch!" << std::endl;
        std::exit(1);
    }
    
    Tensor result(this->h, other.w);
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                this->h, other.w, this->w,
                1.0f,
                this->vals->data(), this->w,
                other.vals->data(), other.w,
                0.0f,
                result.vals->data(), other.w);
    
    return result;
}

Tensor Tensor::createMask(int h, int w){
    if (h != w){
        std::cerr << "Cannot create mask with dimensions: " << h << ", " << w << std::endl;
        std::exit(1);
    }
    
    auto vals = std::make_shared<std::vector<float>>(h * w, -std::numeric_limits<float>::infinity());
    
    for (int i = 0; i < h; i++){
        for (int a = 0; a < w; a++){
            if (a <= i){
                (*vals)[i * w + a] = 0.0f;  // Flat indexing
            }
        }
    }
    
    return Tensor(vals, h, w);
}


void Tensor::randInit(){
    std::random_device rd;
    std::mt19937 gen(rd());
    float scale = std::sqrt(2.0f / (h + w));
    std::normal_distribution<float> dist(0.0f, scale);
    
    for (int i = 0; i < h * w; i++){  // Iterate over flat array
        (*this->vals)[i] = dist(gen);
    }
}

Tensor Tensor::layerNorm(const Tensor& v2) const{
    float theta = 1;
    float beta = 0;
    float eps = 1e-5;
    Tensor result(this->h, this->w);
    
    for (int row = 0; row < this->h; row++){
        float mean = 0;
        for (int col = 0; col < this->w; col++){
            mean += (*this->vals)[row * this->w + col];  // Flat indexing
        }
        mean = mean / this->w;
        
        float variance = 0;
        for (int col = 0; col < this->w; col++){
            float val = (*this->vals)[row * this->w + col];
            variance += std::pow(val - mean, 2);
        }
        variance = variance / this->w;
        
        for (int col = 0; col < this->w; col++){
            float val = (*this->vals)[row * this->w + col];
            (*result.vals)[row * this->w + col] = theta * ((val - mean) / std::sqrt(variance + eps)) + beta;
        }
    }
    return result;
}

void Tensor::transpose(){
    auto newVals = std::make_shared<std::vector<float>>(h * w, 0.0f);
    
    for (int i = 0; i < this->h; i++){
        for (int j = 0; j < this->w; j++){
            (*newVals)[j * this->h + i] = (*this->vals)[i * this->w + j];  // Transpose indexing
        }
    }
    
    this->vals = newVals;
    std::swap(this->h, this->w);
}

std::ostream& operator<<(std::ostream& os, const Tensor& obj){
    os << "[[";
    for (int i = 0; i < obj.h; i++){
        for (int j = 0; j < obj.w; j++){
            if (i > 0 && j == 0){
                os << " ";
            }
            if ((j == obj.w - 1) && (i == obj.h - 1)){
                os << (*obj.vals)[i * obj.w + j] << "]]";  // Flat indexing
            }
            else if (j == obj.w - 1){
                os << (*obj.vals)[i * obj.w + j] << "]";
            }
            else {
                os << (*obj.vals)[i * obj.w + j] << ", ";
            }
        }
        os << std::endl;
    }
    return os;
}

Tensor operator+(const Tensor& a, const Tensor& b){
    if (!(a.w == b.w)){
        std::cerr << "Incorrect shape!" << std::endl;
        std::exit(1);
    } else if ((a.h != b.h) && (b.h == 1)){
        Tensor result(a.h, a.w);
        for (int i = 0; i < a.h; i++){
            for (int j = 0; j < a.w; j++){
                (*result.vals)[i * a.w + j] = (*a.vals)[i * a.w + j] + (*b.vals)[j];  // Flat indexing
            }
        }
        return result;
    } else {
        Tensor result(a.h, a.w);
        for (int i = 0; i < a.h * a.w; i++){  // Simple flat iteration
            (*result.vals)[i] = (*a.vals)[i] + (*b.vals)[i];
        }
        return result;
    }
}


Tensor operator+(const Tensor& a, const float& b){
    Tensor result(a.h, a.w);
    for (int i = 0; i < a.h * a.w; i++){
        (*result.vals)[i] = (*a.vals)[i] + b;
    }
    return result;
}

Tensor operator-(const Tensor& a, const Tensor& b){
    if (a.h != b.h || a.w != b.w){
        std::cerr << "Incorrect shape!" << std::endl;
        std::exit(1);
    }
    Tensor result(a.h, a.w);
    for (int i = 0; i < a.h * a.w; i++){
        (*result.vals)[i] = (*a.vals)[i] - (*b.vals)[i];
    }
    return result;
}

Tensor operator-(const Tensor& a, const float& b){
    Tensor result(a.h, a.w);
    for (int i = 0; i < a.h * a.w; i++){
        (*result.vals)[i] = (*a.vals)[i] - b;
    }
    return result;
}

Tensor operator/(const Tensor& a, const Tensor& b){
    if (a.h != b.h || a.w != b.w){
        std::cerr << "Incorrect shape!" << std::endl;
        std::exit(1);
    }
    Tensor result(a.h, a.w);
    for (int i = 0; i < a.h * a.w; i++){
        if ((*b.vals)[i] == 0){
            std::cerr << "Cannot divide by zero!" << std::endl;
            std::exit(1);
        }
        (*result.vals)[i] = (*a.vals)[i] / (*b.vals)[i];
    }
    return result;
}

Tensor operator/(const Tensor& a, const float& b){
    if (b == 0){
        std::cerr << "Cannot divide by zero!" << std::endl;
        std::exit(1);
    }
    Tensor result(a.h, a.w);
    for (int i = 0; i < a.h * a.w; i++){
        (*result.vals)[i] = (*a.vals)[i] / b;
    }
    return result;
}

Tensor operator*(const Tensor& a, const Tensor& b){
    if (a.h != b.h || a.w != b.w){
        std::cerr << "Incorrect shape!" << std::endl;
        std::exit(1);
    }
    Tensor result(a.h, a.w);
    for (int i = 0; i < a.h * a.w; i++){
        (*result.vals)[i] = (*a.vals)[i] * (*b.vals)[i];
    }
    return result;
}

Tensor operator*(const Tensor& a, const float& b){
    Tensor result(a.h, a.w);
    for (int i = 0; i < a.h * a.w; i++){
        (*result.vals)[i] = (*a.vals)[i] * b;
    }
    return result;
}


std::string Tensor::shape() const{
    return "(" + std::to_string(this->h) + ", " + std::to_string(this->w) + ")";
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
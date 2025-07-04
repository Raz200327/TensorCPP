#ifndef TENSOR
#define TENSOR

#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <string>

class Tensor{
    public:
    std::shared_ptr<std::vector<std::vector<float> > > vals;
    int h;
    int w;
    Tensor(int h, int w);
    Tensor(std::shared_ptr<std::vector<std::vector<float> > > vals, int h, int w);
    Tensor(std::string path, int h, int w);
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& result);
    Tensor& operator=(Tensor&& result) noexcept;
    Tensor matMul(const Tensor &v2) const;
    void randInit();                                
    void transpose();
    static Tensor createMask(int h, int w);
    friend std::ostream& operator<<(std::ostream& os, const Tensor& obj);
    friend Tensor operator+(const Tensor& a, const Tensor& b);
    friend Tensor operator+(const Tensor& a, const float& b);
    friend Tensor operator/(const Tensor& a, const Tensor& b);
    friend Tensor operator/(const Tensor& a, const float& b);
    friend Tensor operator*(const Tensor& a, const Tensor& b);
    friend Tensor operator*(const Tensor& a, const float& b);
    friend Tensor operator-(const Tensor& a, const Tensor& b);
    friend Tensor operator-(const Tensor& a, const float& b);
};

class ActivationFunction{
    public:
    virtual void apply(Tensor &tensor) = 0;
    virtual ~ActivationFunction() = default;
};

class ReLU : public ActivationFunction {
    public:
    void apply(Tensor &tensor) override {
        std::shared_ptr<std::vector<std::vector<float> > > vals = tensor.vals;
        for (int h = 0; h < vals->size(); h++) {
            for (int w = 0; w < (*vals)[h].size(); w++) {
                if ((*vals)[h][w] < 0) {
                    (*vals)[h][w] = 0;
                }
            }
        }
    }
};


class Softmax : public ActivationFunction {
    public:
    void apply(Tensor &tensor) override {
        std::shared_ptr<std::vector<std::vector<float> > > vals = tensor.vals;
        for (int h = 0; h < vals->size(); h++) {
            float denomitatorSum = 0;
            for (int w2 = 0; w2 < (*vals)[h].size(); w2++) {
                denomitatorSum += std::exp((*vals)[h][w2]);
            }
            for (int w = 0; w < (*vals)[h].size(); w++) {
                (*vals)[h][w] = std::exp((*vals)[h][w]) / denomitatorSum;
            } 
        }
    }
};

class CausalSelfAttentionSingleHead {
    public:
    std::unordered_map<std::string, int> config;
    Tensor q_h;
    Tensor k_h;
    Tensor v_h; 
    CausalSelfAttentionSingleHead(const std::unordered_map<std::string, int> &config);
    Tensor forward(const Tensor &input);
};

class NeuralNetwork{
    public:
    std::vector<Tensor> layers;
    std::vector<std::unique_ptr<ActivationFunction> > activationFunctions;
    NeuralNetwork(std::string configFile);
    NeuralNetwork(std::vector<Tensor> layers);
    NeuralNetwork();
    Tensor forward(const Tensor &input);
};

#endif
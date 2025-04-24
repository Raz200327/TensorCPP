#ifndef TENSOR
#define TENSOR

#include <iostream>
#include <vector>
#include <random>
#include <functional>

class Tensor{
    public:
    std::vector<std::vector<float> > *vals;
    int h;
    int w;
    Tensor(int w, int h);
    Tensor(std::vector<std::vector<float> > *vals, int h, int w);
    Tensor matMul(const Tensor &v2);
    void transpose();
    void fillTensor();
    std::string shape() const;
    friend std::ostream& operator<<(std::ostream& os, const Tensor& obj);
    ~Tensor();
};

void ReLU(Tensor &tensor);


class Model{
    public:
    std::vector<Tensor> layers;
};

void loadWeights(Tensor &tensor, std::string file_path);

#endif
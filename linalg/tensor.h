#ifndef TENSOR
#define TENSOR

#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
//7
class Tensor{
    public:
    std::shared_ptr<std::vector<std::vector<float> > > vals;
    int h;
    int w;
    Tensor(int h, int w);
    Tensor(std::shared_ptr<std::vector<std::vector<float> > > vals, int h, int w);
    Tensor(std::string path, int h, int w);
    ~Tensor();
    Tensor& operator=(const Tensor& result);
    Tensor matMul(const Tensor &v2);
    void randInit();
    void ReLU();
    void Softmax();
    friend std::ostream& operator<<(std::ostream& os, const Tensor& obj);
    friend Tensor operator+(const Tensor& a, const Tensor& b);
};

void fillRandMatrix(std::vector<std::vector<float> >& mat, int h, int w);



#endif
#ifndef TENSOR
#define TENSOR

#include <iostream>
#include <vector>
#include <random>

class Tensor{
    public:
    std::vector<std::vector<float> > *vals;
    int h;
    int w;
    Tensor(int w, int h);
    Tensor(std::vector<std::vector<float> > *vals, int h, int w);
    Tensor matMul(const Tensor &v2);
    void transpose();
    friend std::ostream& operator<<(std::ostream& os, const Tensor& obj);
    ~Tensor();
};

void fillMatrix(std::vector<std::vector<float> >& mat, int h, int w);

#endif
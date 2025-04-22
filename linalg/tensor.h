#ifndef TENSOR
#define TENSOR

#include <iostream>
#include <vector>
#include <random>

class Tensor{
    public:
    std::vector<std::vector<float> > vals;
    int h;
    int w;
    Tensor(std::vector<std::vector<float> > vals, int h, int w): vals(vals), h(h), w(w){}
    Tensor matMul(const Tensor &v2);
    friend std::ostream& operator<<(std::ostream& os, const Tensor& obj);
};

void fillMatrix(std::vector<std::vector<float> >& mat, int h, int w);

#endif
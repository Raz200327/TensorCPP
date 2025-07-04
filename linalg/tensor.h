#ifndef TENSOR
#define TENSOR

#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <sstream>
#include <fstream>

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

#endif
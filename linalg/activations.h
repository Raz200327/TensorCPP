#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "tensor.h"

class ActivationFunction{
    public:
    virtual void apply(Tensor &tensor) = 0;
    virtual ~ActivationFunction() = default;
};

class ReLU : public ActivationFunction {
public:
    void apply(Tensor& tensor) override;
};

class Softmax : public ActivationFunction {
public:
    void apply(Tensor& tensor) override;
};

class GELU : public ActivationFunction {
public:
    void apply(Tensor& tensor) override;
};

#endif
#ifndef NEURAL_NET
#define NEURAL_NET

#include "tensor.h"
#include <vector>
#include <memory>               
#include "activations.h"

class NeuralNetwork{
    public:
    float theta;
    float beta;
    float eps;
    std::vector<Tensor> layers;
    std::vector<std::unique_ptr<ActivationFunction> > activationFunctions;
    NeuralNetwork(std::string configFile);
    NeuralNetwork(std::vector<Tensor> layers);
    NeuralNetwork();
    Tensor forward(const Tensor &input);
};

#endif
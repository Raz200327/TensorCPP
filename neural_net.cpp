#include "linalg/neural_net.h"

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
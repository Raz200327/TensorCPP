#include "linalg/activations.h"
#include <cmath>


void ReLU::apply(Tensor& tensor) {
    float* data = tensor.vals->data();
    int size = tensor.h * tensor.w;
    
    for (int i = 0; i < size; i++) {
        if (data[i] < 0) {
            data[i] = 0;
        }
    }
}

void Softmax::apply(Tensor& tensor) {
    float* data = tensor.vals->data();
    
    for (int h = 0; h < tensor.h; h++) {
        int row_start = h * tensor.w;
        
        // Compute sum of exponentials for this row
        float denominatorSum = 0;
        for (int w = 0; w < tensor.w; w++) {
            denominatorSum += std::exp(data[row_start + w]);
        }
        
        // Apply softmax
        for (int w = 0; w < tensor.w; w++) {
            data[row_start + w] = std::exp(data[row_start + w]) / denominatorSum;
        }
    }
}

void GELU::apply(Tensor& tensor) {
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    float* data = tensor.vals->data();
    int size = tensor.h * tensor.w;
    
    for (int i = 0; i < size; i++) {
        float x = data[i];
        float cubic_term = 0.044715f * std::pow(x, 3);
        float tanh_input = sqrt_2_over_pi * (x + cubic_term);
        data[i] = 0.5f * x * (1.0f + std::tanh(tanh_input));
    }
}
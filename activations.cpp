#include "linalg/activations.h"


void ReLU::apply(Tensor &tensor) {
    std::shared_ptr<std::vector<std::vector<float> > > vals = tensor.vals;
    for (int h = 0; h < vals->size(); h++) {
        for (int w = 0; w < (*vals)[h].size(); w++) {
            if ((*vals)[h][w] < 0) {
                (*vals)[h][w] = 0;
            }
        }
    }
}

void Softmax::apply(Tensor &tensor) {
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

void GELU::apply(Tensor &tensor) {
    std::shared_ptr<std::vector<std::vector<float> > > vals = tensor.vals;
    for (int h = 0; h < vals->size(); h++) {
        for (int w = 0; w < (*vals)[h].size(); w++) {
            (*vals)[h][w] = 0.5 * (*vals)[h][w] * (1 + std::tanh(std::sqrt(2 / M_PI) * ((*vals)[h][w] + 0.044715 * std::pow((*vals)[h][w], 3))));
        }
    }
}

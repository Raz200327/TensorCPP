#ifndef TRANSFORMER
#define TRANSFORMER

#include "tensor.h"
#include <unordered_map>

class CausalSelfAttentionSingleHead {
    public:
    std::unordered_map<std::string, int> config;
    Tensor q_h;
    Tensor k_h;
    Tensor v_h; 
    CausalSelfAttentionSingleHead(const std::unordered_map<std::string, int> &config);
    Tensor forward(const Tensor &input);
};

#endif
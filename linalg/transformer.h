#ifndef TRANSFORMER
#define TRANSFORMER

#include "tensor.h"
#include "activations.h"
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

class CausalSelfAttention {
    public:
    Tensor c_proj;
    std::unordered_map<std::string, int> config;
    std::vector<CausalSelfAttentionSingleHead> heads;
    CausalSelfAttention(const std::unordered_map<std::string, int> &config);
    Tensor forward(const Tensor &input);
};

class MLP {
    public:
    std::unordered_map<std::string, int> config;
    Tensor c_fc;
    Tensor c_proj;
    GELU gelu;
    MLP(const std::unordered_map<std::string, int> &config);
    Tensor forward(const Tensor &input);
};


#endif
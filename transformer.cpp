#include "linalg/transformer.h"
#include "linalg/activations.h"

CausalSelfAttentionSingleHead::CausalSelfAttentionSingleHead(const std::unordered_map<std::string, int> &config)
    : q_h(Tensor(config.at("n_emb"), config.at("n_emb")/config.at("n_heads"))),
      k_h(Tensor(config.at("n_emb"), config.at("n_emb")/config.at("n_heads"))),
      v_h(Tensor(config.at("n_emb"), config.at("n_emb")/config.at("n_heads"))),
      config(config) {
    this->q_h.randInit();
    this->k_h.randInit();
    this->v_h.randInit();
}

Tensor CausalSelfAttentionSingleHead::forward(const Tensor &input){

    Tensor q = input.matMul(this->q_h);
    Tensor k = input.matMul(this->k_h);
    Tensor v = input.matMul(this->v_h);
    Tensor mask = Tensor::createMask(input.h, input.h);
    k.transpose();
    Tensor att = q.matMul(k);
    att = att / std::sqrt(this->config.at("n_emb") / this->config.at("n_heads"));
    att = att + mask;
    Softmax activation;
    activation.apply(att);
    Tensor result = att.matMul(v);
    return result;
}

CausalSelfAttention::CausalSelfAttention(const std::unordered_map<std::string, int> &config): c_proj(Tensor(config.at("n_emb"), config.at("n_emb"))){
    this->config = config;
    for (int i = 0; i < config.at("n_heads"); i++){
        this->heads.push_back(CausalSelfAttentionSingleHead(config));
    }
    this->c_proj.randInit();
}

Tensor CausalSelfAttention::forward(const Tensor &input){
    Tensor result(input.h, 0);
    for (int i = 0; i < this->config.at("n_heads"); i++){
        result.concat(this->heads[i].forward(input));
    }
    result = result.matMul(this->c_proj);
    return result;
}

MLP::MLP(const std::unordered_map<std::string, int> &config)
    : c_fc(Tensor(config.at("n_emb"), config.at("n_emb")*4)),
      c_proj(Tensor(config.at("n_emb")*4, config.at("n_emb"))),
      gelu(GELU()) {
    this->c_fc.randInit();
    this->c_proj.randInit();
}

Tensor MLP::forward(const Tensor &input){
    Tensor result = input.matMul(this->c_fc);
    this->gelu.apply(result);
    result = result.matMul(this->c_proj);
    return result;
}
#include "linalg/transformer.h"
#include "linalg/activations.h"

CausalSelfAttentionSingleHead::CausalSelfAttentionSingleHead(const std::unordered_map<std::string, int> &config)
    : q_h(Tensor(config.at("n_emb_h"), config.at("n_emb_h"))),
      k_h(Tensor(config.at("n_emb_h"), config.at("n_emb_h"))),
      v_h(Tensor(config.at("n_emb_h"), config.at("n_emb_h"))),
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
    std::cout << "Mask:" << std::endl;
    std::cout << mask << std::endl;
    k.transpose();
    Tensor att = q.matMul(k);
    att = att / std::sqrt(this->config.at("n_emb_h"));
    att = att + mask;
    std::cout << "Att:" << std::endl;
    std::cout << att << std::endl;
    Softmax activation;
    activation.apply(att);
    std::cout << "Att after softmax:" << std::endl;
    std::cout << att << std::endl;
    Tensor result = att.matMul(v);
    return result;
}
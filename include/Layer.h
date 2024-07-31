#ifndef LAYER_H
#define LAYER_H

#include <memory>
#include "Macro.h"
#include "Tensor.h"

class Layer {
    private:
        std::shared_ptr<Tensor> preNorm;
        std::shared_ptr<Tensor> postNorm;
        std::shared_ptr<Tensor> mlpGate;
        std::shared_ptr<Tensor> mlpUp;
        std::shared_ptr<Tensor> mlpDown;
        std::shared_ptr<Tensor> queryWeight;
        std::shared_ptr<Tensor> queryBias;
        std::shared_ptr<Tensor> keyWeight;
        std::shared_ptr<Tensor> keyBias;
        std::shared_ptr<Tensor> valueWeight;
        std::shared_ptr<Tensor> valueBias;
        std::shared_ptr<Tensor> outputLinear;
    public:
        Layer();
        ~Layer() = default;
        Tensor * input_layernorm() {
            return preNorm.get();
        }
        Tensor * post_attention_layernorm() {
            return postNorm.get();
        }
        Tensor * mlp_gate() {
            return mlpGate.get();
        }
        Tensor * mlp_up() {
            return mlpUp.get();
        }
        Tensor * mlp_down() {
            return mlpDown.get();
        }
        Tensor * query_weight() {
            return queryWeight.get();
        }
        Tensor * query_bias() {
            return queryBias.get();
        }
        Tensor * key_weight() {
            return keyWeight.get();
        }
        Tensor * key_bias() {
            return keyBias.get();
        }
        Tensor * value_weight() {
            return valueWeight.get();
        }
        Tensor * value_bias() {
            return valueBias.get();
        }
        Tensor * attention_out_linear() {
            return outputLinear.get();
        }
        
};

#endif
#ifndef LAYER_H
#define LAYER_H

#include <memory>
#include "Macro.h"
#include "Tensor.h"
#include "Norm.h"
#include "Attention.h"
#include "Linear.h"
#include "MLP.h"

class Layer {
    private:
        int layerID;
        int hidden_size;
        int intermediate_size;
        int num_head;
        int num_kvhead;
        int head_dim;
        std::shared_ptr<Norm> input_layer_norm;
        std::shared_ptr<Linear> query_linear;
        std::shared_ptr<Linear> key_linear;
        std::shared_ptr<Linear> value_linear;
        std::shared_ptr<Attention> attention;
        std::shared_ptr<Linear> post_attention_linear;
        std::shared_ptr<Norm> post_attention__norm;
        std::shared_ptr<MLP> mlp;
    public:
        Layer(int id, int hiddenSize, int interSize, int numHead, int numKVhead, int headDim);
        ~Layer() = default;
        void load(Loader * loader, size_t offset);
        Tensor forward(Tensor input);
};

#endif
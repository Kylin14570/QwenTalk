#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include "Macro.h"
#include "Tensor.h"
#include "Layer.h"
#include "Loader.h"

class Model {
    public:
        struct ModelConfig{
            int num_layer;
            int num_head;
            int num_kvhead;
            int head_dim;
            int hidden_size;
            int intermediate_size;
            int vocabulary_size;
            int rms_eps;
        };
    private:
        ModelConfig mConfig;
        std::shared_ptr<Tensor> token_embeds;
        std::vector< std::shared_ptr<Layer> > layers;
        std::shared_ptr<Norm> norm;
    public:
        Model(ModelConfig config);
        ~Model() = default;
        void load(const char * model_file_path);
        Tensor forward(Tensor input_embeds);
};

#endif
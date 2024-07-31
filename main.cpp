#include <cstdio>
#include <iostream>
#include <memory>
#include "Macro.h"
#include "BF16.h"
#include "Tensor.h"
#include "Loader.h"
#include "Model.h"

int main(int argc, char * argv[])
{
    if (argc != 2) {
        printf("Usage: %s [Model File Path]\n", argv[0]);
        return 0;
    }
    Model::ModelConfig config;
    config.head_dim = HeadDim;
    config.hidden_size = HiddenSize;
    config.intermediate_size = InterSize;
    config.num_head = NumHead;
    config.num_kvhead = NumKvHead;
    config.num_layer = NumLayer;
    config.rms_eps = RMS_EPS;
    config.vocabulary_size = VocabSize;
    Model * model = new Model(config);
    model->load(argv[1]);
    delete model;
    return 0;
}
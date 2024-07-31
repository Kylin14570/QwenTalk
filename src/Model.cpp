#include "Model.h"
#include "BF16.h"

Model::Model(ModelConfig config) {
    mConfig = config;
    token_embeds.reset(new Tensor(config.vocabulary_size, config.hidden_size));
    norm.reset(new Norm(config.hidden_size, config.rms_eps));
    layers.resize(config.num_layer);
    for (int i = 0; i < config.num_layer; i++) {
        layers[i].reset(
            new Layer(
                i,
                config.hidden_size,
                config.intermediate_size,
                config.num_head,
                config.num_kvhead,
                config.head_dim
            )
        );
    }
}

void Model::load(const char * model_file_path)
{
    printf("Start to load the model from %s\n", model_file_path);
    size_t layerSize = 0;
    layerSize += 2UL * HiddenSize;
    layerSize += 3UL * HiddenSize * InterSize;
    layerSize += 2UL * NumKvHead * HeadDim * (HiddenSize + 1);
    layerSize += NumHead * HeadDim * (HiddenSize + 1);
    layerSize += HiddenSize * HiddenSize;
    layerSize *= 2UL;
    std::shared_ptr<Loader> loader(new Loader(model_file_path));
    size_t offset = HEADER_SIZE;
    printf("Loading the token embeds ...\n");
    loader->load(offset, token_embeds.get());
    offset += token_embeds->size() * sizeof(bf16_t);
    printf("Done!\n");
    for (int i = 0; i < mConfig.num_layer; i++) {
        printf("Loading layer %d ...\n", i);
        layers[i]->load(loader.get(), offset);
        offset += layerSize;
        printf("Done!\n");
    }
    printf("Loading the normalization weight ...\n");
    norm->load(loader.get(), offset);
    printf("Done!\n");
    PRINT_SUCCESS("Model has been loaded!\n");
}

Tensor Model::forward(Tensor input_embeds)
{
    return input_embeds;
}
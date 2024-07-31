#include "Loader.h"
#include "BF16.h"
#include "Buffer.h"

Loader::Loader(const char * path)
{
    fp = fopen(path, "rb");
    if (fp == NULL) {
        printf("Failed to open the file: %s", path);
        exit(0);
    }
}

Loader::~Loader()
{
    if (fp != NULL) {
        fclose(fp);
    }
}

size_t Loader::load(size_t offset, Tensor * tensor)
{
    std::shared_ptr<Buffer> buf(new Buffer(tensor->size() * sizeof(bf16_t)));
    fseek(fp, offset, SEEK_SET);
    size_t readbytes = fread(buf->addr(), 1, tensor->size() * sizeof(bf16_t), fp);
    bf16_t * p = (bf16_t *)buf->addr();
    if (tensor->dim() == 1) {
        for (int i = 0; i < tensor->shape().width; i++){
            tensor->at(i) = bf16_to_float(p[i]);
        }
    }
    else if(tensor->dim() == 2) {
        for (int i = 0; i < tensor->shape().height; i++) {
            for (int j = 0; j < tensor->shape().width; j++) {
                tensor->at(i, j) = bf16_to_float(p[i * tensor->shape().width + j]);
            }
        }
    }
    else if(tensor->dim() == 3) {
        int C = tensor->shape().channel;
        int H = tensor->shape().height;
        int W = tensor->shape().width;
        for (int i = 0; i < C; i++) {
            for (int j = 0; j < H; j++) {
                for (int k = 0; k < W; k++) {
                    tensor->at(i, j, k) = bf16_to_float(p[i * H * W + j * W + k]);
                }
            }
        }
    }
    else {
        int N = tensor->shape().batch;
        int C = tensor->shape().channel;
        int H = tensor->shape().height;
        int W = tensor->shape().width;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < C; j++) {
                for (int k = 0; k < H; k++) {
                    for (int s = 0; s < W; s++) {
                        tensor->at(i, j, k, s) = bf16_to_float(p[i * C * H * W + j * H * W + k * W + s]);
                    }
                }
            }
        }
    }
    return readbytes;
}

size_t Loader::load_embed(Tensor * tensor)
{
    bf16_t * buf = (bf16_t *)malloc(VocabSize * HiddenSize * 2UL);
    fseek(fp, HEADER_SIZE, SEEK_SET);
    size_t readsize = fread(buf, 1, VocabSize * HiddenSize * 2UL, fp);
    for (int i = 0; i < VocabSize; i++) {
        for (int j = 0; j < HiddenSize; j++) {
            tensor->at(i, j) = bf16_to_float(buf[i * HiddenSize + j]);
        }
    }
    free(buf);
    return readsize;
}

size_t Loader::load_norm(Tensor * tensor)
{
    size_t layerSize = 0;
    layerSize += 2UL * HiddenSize;
    layerSize += 3UL * HiddenSize * InterSize;
    layerSize += 2UL * NumKvHead * HeadDim * (HiddenSize + 1);
    layerSize += NumHead * HeadDim * (HiddenSize + 1);
    layerSize += HiddenSize * HiddenSize;
    layerSize *= 2UL;
    fseek(fp, HEADER_SIZE + NumLayer * layerSize, SEEK_SET);
    bf16_t * buf = (bf16_t *)malloc(HiddenSize * 2UL);
    size_t readsize = fread(buf, 1, HiddenSize * 2UL, fp);
    for (int i = 0; i < HiddenSize; i++) {
        tensor->at(i) = bf16_to_float(buf[i]);
    }
    free(buf);
    return readsize;
}

size_t Loader::load_layer(Layer * layer, int k)
{
    size_t layerSize = 0;
    layerSize += 2UL * HiddenSize;
    layerSize += 3UL * HiddenSize * InterSize;
    layerSize += 2UL * NumKvHead * HeadDim * (HiddenSize + 1);
    layerSize += NumHead * HeadDim * (HiddenSize + 1);
    layerSize += HiddenSize * HiddenSize;
    layerSize *= 2UL;
    fseek(fp, HEADER_SIZE + k * layerSize, SEEK_SET);
    size_t readsize = 0;
    bf16_t * buf = (bf16_t *)malloc(InterSize * HiddenSize * 2UL);

    // input_layernorm.weight
    readsize += fread(buf, 1, HiddenSize * 2UL, fp);
    for (int i = 0; i < HiddenSize; i++) {
        layer->input_layernorm()->at(i) = bf16_to_float(buf[i]);
    }

    // mlp.gate_proj.weight
    readsize += fread(buf, 1, InterSize * HiddenSize * 2UL, fp);
    for (int i = 0; i < InterSize; i++) {
        for (int j = 0; j < HiddenSize; j++) {
            layer->mlp_gate()->at(i, j) = bf16_to_float(buf[i * HiddenSize + j]);
        }
    }

    // mlp.up_proj.weight
    readsize += fread(buf, 1, InterSize * HiddenSize * 2UL, fp);
    for (int i = 0; i < InterSize; i++) {
        for (int j = 0; j < HiddenSize; j++) {
            layer->mlp_up()->at(i, j) = bf16_to_float(buf[i * HiddenSize + j]);
        }
    }

    // mlp.down_proj.weight
    readsize += fread(buf, 1, HiddenSize * InterSize * 2UL, fp);
    for (int i = 0; i < HiddenSize; i++) {
        for (int j = 0; j < InterSize; j++) {
            layer->mlp_down()->at(i, j) = bf16_to_float(buf[i * InterSize + j]);
        }
    }

    // post_attention_layernorm.weight
    readsize += fread(buf, 1, HiddenSize * 2UL, fp);
    for (int i = 0; i < HiddenSize; i++) {
        layer->post_attention_layernorm()->at(i) = bf16_to_float(buf[i]);
    }

    // self_attn.k_proj.bias
    readsize += fread(buf, 1, NumKvHead * HeadDim * 2UL, fp);
    for (int i = 0; i < NumKvHead * HeadDim; i++) {
        layer->key_bias()->at(i) = bf16_to_float(buf[i]);
    }

    // self_attn.k_proj.weight
    readsize += fread(buf, 1, NumKvHead * HeadDim * HiddenSize * 2UL, fp);
    for (int i = 0; i < NumKvHead * HeadDim; i++) {
        for (int j = 0; j < HiddenSize; j++) {
            layer->key_weight()->at(i, j) = bf16_to_float(buf[i * HiddenSize + j]);
        }
    }

    // self_attn.o_proj.weight
    readsize += fread(buf, 1, HiddenSize * HiddenSize * 2UL, fp);
    for (int i = 0; i < HiddenSize; i++) {
        for (int j = 0; j < HiddenSize; j++) {
            layer->attention_out_linear()->at(i, j) = bf16_to_float(buf[i * HiddenSize + j]);
        }
    }

    // self_attn.q_proj.bias
    readsize += fread(buf, 1, NumHead * HeadDim * 2UL, fp);
    for (int i = 0; i < NumHead * HeadDim; i++) {
        layer->query_bias()->at(i) = bf16_to_float(buf[i]);
    }

    // self_attn.q_proj.weight
    readsize += fread(buf, 1, NumHead * HeadDim * HiddenSize * 2UL, fp);
    for (int i = 0; i < NumHead * HeadDim; i++) {
        for (int j = 0; j < HiddenSize; j++) {
            layer->query_weight()->at(i, j) = bf16_to_float(buf[i * HiddenSize + j]);
        }
    }

    // self_attn.v_proj.bias
    readsize += fread(buf, 1, NumKvHead * HeadDim * 2UL, fp);
    for (int i = 0; i < NumKvHead * HeadDim; i++) {
        layer->value_bias()->at(i) = bf16_to_float(buf[i]);
    }

    // self_attn.v_proj.weight
    readsize += fread(buf, 1, NumKvHead * HeadDim * HiddenSize * 2UL, fp);
    for (int i = 0; i < NumKvHead * HeadDim; i++) {
        for (int j = 0; j < HiddenSize; j++) {
            layer->value_weight()->at(i, j) = bf16_to_float(buf[i * HiddenSize + j]);
        }
    }

    free(buf);
    return readsize;
}
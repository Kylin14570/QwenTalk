#include "Loader.h"
#include "BF16.h"

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

size_t Loader::load_embed(Tensor * tensor)
{
    bf16_t * buf = (bf16_t *)malloc(VocabSize * HiddenSize * 2UL);
    if (buf == NULL) {
        printf("Failed to allocate memory in load_embed()\n");
        exit(0);
    }
    fseek(fp, HEADER_SIZE, SEEK_SET);
    size_t readsize = fread(buf, VocabSize * HiddenSize * 2UL, 1, fp);
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
    if (buf == NULL) {
        printf("Failed to allocate memory in load_embed()\n");
        exit(0);
    }
    size_t readsize = fread(buf, HiddenSize * 2UL, 1, fp);
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
    
    return 0;
}
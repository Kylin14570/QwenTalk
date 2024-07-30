#include "Loader.h"

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
    fseek(fp, HEADER_SIZE, SEEK_SET);
    return fread(tensor->host(), VocabSize * HiddenSize, 1, fp);
}

size_t Loader::load_layer(Layer * layer, int k)
{
    fseek(fp, HEADER_SIZE + k * LAYER_SIZE, SEEK_SET);
    return 0;
}

size_t Loader::load_norm(Tensor * tensor)
{
    fseek(fp, HEADER_SIZE + NumLayer * LAYER_SIZE, SEEK_SET);
    return 0;
}
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

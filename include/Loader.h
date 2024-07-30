#ifndef LOADER_H
#define LOADER_H

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include "Macro.h"
#include "Tensor.h"
#include "Layer.h"

class Loader {
    private:
        FILE *fp = NULL;
    public:
        Loader(const char * path);
        ~Loader();
        size_t load_embed(Tensor * tensor);
        size_t load_layer(Layer * layer, int k);
        size_t load_norm(Tensor * tensor);
};

#endif
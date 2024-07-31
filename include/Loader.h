#ifndef LOADER_H
#define LOADER_H

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include "Macro.h"
#include "Tensor.h"

class Loader {
    private:
        FILE *fp = NULL;
    public:
        Loader(const char * path);
        ~Loader();
        size_t load(size_t offset, Tensor * tensor);
};

#endif
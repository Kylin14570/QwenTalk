#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <cassert>
#include <memory>
#include "Buffer.h"
#include "Malloc.h"

struct Shape {
    int batch;
    int channel;
    int height;
    int width;
};

class Tensor {
    private:
        int Tdim;
        Shape Tshape;
        size_t Tsize;
        std::shared_ptr<Buffer> Tbuf;
    public:
        Tensor();
        Tensor(int W);
        Tensor(int H, int W);
        Tensor(int C, int H, int W);
        Tensor(int N, int C, int H, int W);
        Tensor(Shape _shape);
        ~Tensor();
        Tensor(const Tensor & src);
        Tensor & operator= (const Tensor & src);
        char * host();
        size_t size();
        int dim();
        Shape shape();
        float & at(int i);
        float & at(int i, int j);
        float & at(int i, int j, int k);
        float & at(int i, int j, int k, int s);
        void print();
};

#endif
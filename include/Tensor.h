#ifndef TENSOR_H
#define TENSOR_H

#include <cstddef>
#include <cassert>

struct Shape {
    int batch;
    int channel;
    int height;
    int width;
};

class Tensor {
    private:
        int dim;
        Shape Tshape;
        size_t Tsize;
        float * Tbuf;
    public:
        Tensor();
        Tensor(int W);
        Tensor(int H, int W);
        Tensor(int C, int H, int W);
        Tensor(int N, int C, int H, int W);
        Tensor(Shape _shape);
        ~Tensor();
        float * host();
        Shape shape();
        float & at(int i);
        float & at(int i, int j);
        float & at(int i, int j, int k);
        float & at(int i, int j, int k, int s);
        void print();
};

#endif
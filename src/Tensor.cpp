#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "Tensor.h"

Tensor::Tensor()
{
    dim = 0;
    Tshape.batch   = 1;
    Tshape.channel = 1;
    Tshape.height  = 1;
    Tshape.width   = 1;
    Tsize = 1;
    Tbuf.reset(new Buffer(Tsize * sizeof(float)));
}

Tensor::Tensor(int W)
{
    assert(W > 0);
    dim = 1;
    Tshape.batch   = 1;
    Tshape.channel = 1;
    Tshape.height  = 1;
    Tshape.width   = W;
    Tsize = W;
    Tbuf.reset(new Buffer(Tsize * sizeof(float)));
}

Tensor::Tensor(int H, int W)
{
    assert(H > 0 && W > 0);
    dim = 2;
    Tshape.batch   = 1;
    Tshape.channel = 1;
    Tshape.height  = H;
    Tshape.width   = W;
    Tsize = H * W;
    Tbuf.reset(new Buffer(Tsize * sizeof(float)));
}

Tensor::Tensor(int C, int H, int W)
{
    assert(C > 0 && H > 0 && W > 0);
    dim = 3;
    Tshape.batch   = 1;
    Tshape.channel = C;
    Tshape.height  = H;
    Tshape.width   = W;
    Tsize = C * H * W;
    Tbuf.reset(new Buffer(Tsize * sizeof(float)));
}

Tensor::Tensor(int N, int C, int H, int W)
{
    assert(N > 0 && C > 0 && H > 0 && W > 0);
    dim = 4;
    Tshape.batch   = N;
    Tshape.channel = C;
    Tshape.height  = H;
    Tshape.width   = W;
    Tsize = N * C * H * W;
    Tbuf.reset(new Buffer(Tsize * sizeof(float)));
}

Tensor::Tensor(Shape _shape)
{
    assert(_shape.batch > 0);
    assert(_shape.channel > 0);
    assert(_shape.height > 0);
    assert(_shape.width > 0);
    dim = 4;
    Tshape = _shape;
    Tsize = Tshape.batch * Tshape.channel * Tshape.height * Tshape.width;
    Tbuf.reset(new Buffer(Tsize * sizeof(float)));
}

Tensor::~Tensor()
{
    Tbuf.reset();
    Tsize = 0;
}

Tensor::Tensor(const Tensor & src)
{
    this->Tshape = src.Tshape;
    this->Tsize = src.Tsize;
    this->dim = src.dim;
    this->Tbuf = src.Tbuf;
}

Tensor & Tensor::operator= (const Tensor & src)
{
    if (&src == this) {
        return *this;
    }
    this->Tshape = src.Tshape;
    this->Tsize = src.Tsize;
    this->dim = src.dim;
    this->Tbuf = src.Tbuf;
    return *this;
}

float * Tensor::host()
{
    return (float *)Tbuf->addr();
}

Shape Tensor::shape()
{
    return Tshape;
}

float & Tensor::at(int i)
{
    assert(dim == 1);
    return host()[i];
}

float & Tensor::at(int i, int j)
{
    assert(dim == 2);
    int W = Tshape.width;
    return host()[i * W + j];
}

float & Tensor::at(int i, int j, int k)
{
    assert(dim == 3);
    int H = Tshape.height;
    int W = Tshape.width;
    return host()[i * H * W + j * W + k];
}

float & Tensor::at(int i, int j, int k, int s)
{
    assert(dim == 4);
    int C = Tshape.channel;
    int H = Tshape.height;
    int W = Tshape.width;
    return host()[i * C * H * W + j * H * W + k * W + s];
}

void Tensor::print()
{
    if (dim == 0) {
        printf("%f\n", host()[0]);
    }
    else if (dim == 1) {
        printf("(");
        for (int i = 0; i < Tshape.width; i++) {
            printf("%f", at(i));
            if (i < Tshape.width - 1)
                printf(", ");
        }
        printf(")\n");
    }
    else if (dim == 2) {
        printf("[");
        for (int i = 0; i < Tshape.height; i++) {
            printf("(");
            for (int j = 0; j < Tshape.width; j++) {
                printf("%f", at(i, j));
                if (j < Tshape.width - 1)
                    printf(", ");
            }
            printf(")");
            if (i < Tshape.height - 1)
                printf(", ");
        }
        printf("]\n");
    }
    else if (dim == 3) {
        printf("{");
        for (int i = 0; i < Tshape.channel; i++) {
            printf("[");
            for (int j = 0; j < Tshape.height; j++) {
                printf("(");
                for (int k = 0; k < Tshape.width; k++) {
                    printf("%f", at(i, j, k));
                    if (k < Tshape.width - 1)
                        printf(", ");
                }
                printf(")");
                if (j < Tshape.height - 1)
                    printf(", ");
            }
            printf("]");
            if (i < Tshape.channel - 1)
                printf(", ");
        }
        printf("}\n");
    }
    else {
        printf("TODO\n");
    }
}
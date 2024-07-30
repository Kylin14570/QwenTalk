#include <cstdio>
#include <cstdlib>
#include "Tensor.h"

Tensor::Tensor()
{
    dim = 0;
    Tshape.batch   = 1;
    Tshape.channel = 1;
    Tshape.height  = 1;
    Tshape.width   = 1;
    Tsize = 1;
    Tbuf = (float *)malloc(Tsize * sizeof(float));
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
    Tbuf = (float *)malloc(Tsize * sizeof(float));
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
    Tbuf = (float *)malloc(Tsize * sizeof(float));
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
    Tbuf = (float *)malloc(Tsize * sizeof(float));
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
    Tbuf = (float *)malloc(Tsize * sizeof(float));
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
    Tbuf = (float *)malloc(Tsize * sizeof(float));
}

Tensor::~Tensor()
{
    if (Tbuf) {
        free(Tbuf);
    }
    Tsize = 0;
}

float * Tensor::host()
{
    return Tbuf;
}

Shape Tensor::shape()
{
    return Tshape;
}

float & Tensor::at(int i)
{
    assert(dim == 1);
    return Tbuf[i];
}

float & Tensor::at(int i, int j)
{
    assert(dim == 2);
    int W = Tshape.width;
    return Tbuf[i * W + j];
}

float & Tensor::at(int i, int j, int k)
{
    assert(dim == 3);
    int H = Tshape.height;
    int W = Tshape.width;
    return Tbuf[i * H * W + j * W + k];
}

float & Tensor::at(int i, int j, int k, int s)
{
    assert(dim == 4);
    int C = Tshape.channel;
    int H = Tshape.height;
    int W = Tshape.width;
    return Tbuf[i * C * H * W + j * H * W + k * W + s];
}

void Tensor::print()
{
    if (dim == 0) {
        printf("%f\n", Tbuf[0]);
    }
    else if (dim == 1) {
        printf("[");
        for (int i = 0; i < Tshape.width; i++)
            printf("%f%s", Tbuf[i], (i < Tshape.width - 1) ? ", " : "]\n");
    }
    else if (dim == 2) {
        for (int i = 0; i < Tshape.height; i++) {
            printf("[");
            for (int j = 0; j < Tshape.width; j++)
                printf("%f%s", at(i, j), (j < Tshape.width - 1) ? ", " : "]\n");
        }
    }
    else if (dim == 3) {
        printf("TODO\n");
    }
    else {
        printf("TODO\n");
    }
}
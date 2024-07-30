#include <cstdlib>
#include "Tensor.h"

Tensor::Tensor()
{
    Tshape.batch   = 1;
    Tshape.channel = 1;
    Tshape.height  = 1;
    Tshape.width   = 1;
    Tsize = 1;
    Tbuf = (float *)malloc(Tsize * sizeof(float));
}

Tensor::Tensor(int W)
{
    Tshape.batch   = 1;
    Tshape.channel = 1;
    Tshape.height  = 1;
    Tshape.width   = W;
    Tsize = W;
    Tbuf = (float *)malloc(Tsize * sizeof(float));
}

Tensor::Tensor(int H, int W)
{
    Tshape.batch   = 1;
    Tshape.channel = 1;
    Tshape.height  = H;
    Tshape.width   = W;
    Tsize = H * W;
    Tbuf = (float *)malloc(Tsize * sizeof(float));
}

Tensor::Tensor(int C, int H, int W)
{
    Tshape.batch   = 1;
    Tshape.channel = C;
    Tshape.height  = H;
    Tshape.width   = W;
    Tsize = C * H * W;
    Tbuf = (float *)malloc(Tsize * sizeof(float));
}

Tensor::Tensor(int N, int C, int H, int W)
{
    Tshape.batch   = N;
    Tshape.channel = C;
    Tshape.height  = H;
    Tshape.width   = W;
    Tsize = N * C * H * W;
    Tbuf = (float *)malloc(Tsize * sizeof(float));
}

Tensor::Tensor(Shape _shape)
{
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

float Tensor::at(int i)
{
    return Tbuf[i];
}

float Tensor::at(int i, int j)
{
    int W = Tshape.width;
    return Tbuf[i * W + j];
}

float Tensor::at(int i, int j, int k)
{
    int H = Tshape.height;
    int W = Tshape.width;
    return Tbuf[i * H * W + j * W + k];
}

float Tensor::at(int i, int j, int k, int s)
{
    int C = Tshape.channel;
    int H = Tshape.height;
    int W = Tshape.width;
    return Tbuf[i * C * H * W + j * H * W + k * W + s];
}
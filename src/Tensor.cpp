#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "Tensor.h"

Tensor::Tensor()
{
    Tdim = 0;
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
    Tdim = 1;
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
    Tdim = 2;
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
    Tdim = 3;
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
    Tdim = 4;
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
    Tdim = 4;
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
    this->Tsize  = src.Tsize;
    this->Tdim   = src.Tdim;
    this->Tbuf   = src.Tbuf;
}

Tensor & Tensor::operator= (const Tensor & src)
{
    if (&src == this) {
        return *this;
    }
    this->Tshape = src.Tshape;
    this->Tsize  = src.Tsize;
    this->Tdim   = src.Tdim;
    this->Tbuf   = src.Tbuf;
    return *this;
}

char * Tensor::host()
{
    return Tbuf->addr();
}

size_t Tensor::size()
{
    return Tsize;
}

int Tensor::dim()
{
    return Tdim;
}

Shape Tensor::shape()
{
    return Tshape;
}

float & Tensor::at(int i)
{
    assert(Tdim == 1);
    return *((float *)host() + i);
}

float & Tensor::at(int i, int j)
{
    assert(Tdim == 2);
    int W = Tshape.width;
    return *((float *)host() + i * W + j);
}

float & Tensor::at(int i, int j, int k)
{
    assert(Tdim == 3);
    int H = Tshape.height;
    int W = Tshape.width;
    return *((float *)host() + i * H * W + j * W + k);
}

float & Tensor::at(int i, int j, int k, int s)
{
    assert(Tdim == 4);
    int C = Tshape.channel;
    int H = Tshape.height;
    int W = Tshape.width;
    return *((float *)host() + i * C * H * W + j * H * W + k * W + s);
}

void Tensor::print()
{
    if (Tdim == 0) {
        printf("%f\n", *(float *)host());
    }
    else if (Tdim == 1) {
        printf("(");
        for (int i = 0; i < Tshape.width; i++) {
            printf("%f", at(i));
            if (i < Tshape.width - 1)
                printf(", ");
        }
        printf(")\n");
    }
    else if (Tdim == 2) {
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
    else if (Tdim == 3) {
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

Tensor Tensor::operator+ (const Tensor & tadd)
{
    bool shapeSame = true;
    shapeSame = shapeSame && (this->Tdim == tadd.Tdim);
    shapeSame = shapeSame && (this->Tshape.batch == tadd.Tshape.batch);
    shapeSame = shapeSame && (this->Tshape.channel == tadd.Tshape.channel);
    shapeSame = shapeSame && (this->Tshape.height == tadd.Tshape.height);
    shapeSame = shapeSame && (this->Tshape.width == tadd.Tshape.width);
    if (!shapeSame) {
        PRINT_ERROR("Cannot add tensors with different shapes!\n");
        return *this;
    }
    Tensor sumT(this->shape());
    float * ps = (float *)sumT.Tbuf->addr();
    float * p1 = (float *)this->Tbuf->addr();
    float * p2 = (float *)tadd.Tbuf->addr();
    for (int i = 0; i < this->Tsize; i++) {
        ps[i] = p1[i] + p2[i];
    }
    return sumT;
}

Tensor Tensor::operator- (const Tensor & tsub)
{
    bool shapeSame = true;
    shapeSame = shapeSame && (this->Tdim == tsub.Tdim);
    shapeSame = shapeSame && (this->Tshape.batch == tsub.Tshape.batch);
    shapeSame = shapeSame && (this->Tshape.channel == tsub.Tshape.channel);
    shapeSame = shapeSame && (this->Tshape.height == tsub.Tshape.height);
    shapeSame = shapeSame && (this->Tshape.width == tsub.Tshape.width);
    if (!shapeSame) {
        PRINT_ERROR("Cannot add tensors with different shapes!\n");
        return *this;
    }
    Tensor dif(this->shape());
    float * ps = (float *)dif.Tbuf->addr();
    float * p1 = (float *)this->Tbuf->addr();
    float * p2 = (float *)tsub.Tbuf->addr();
    for (int i = 0; i < this->Tsize; i++) {
        ps[i] = p1[i] - p2[i];
    }
    return dif;
}

Tensor Tensor::operator* (const Tensor & tmul)
{
    bool shapeSame = true;
    shapeSame = shapeSame && (this->Tdim == tmul.Tdim);
    shapeSame = shapeSame && (this->Tshape.batch == tmul.Tshape.batch);
    shapeSame = shapeSame && (this->Tshape.channel == tmul.Tshape.channel);
    shapeSame = shapeSame && (this->Tshape.height == tmul.Tshape.height);
    shapeSame = shapeSame && (this->Tshape.width == tmul.Tshape.width);
    if (!shapeSame) {
        PRINT_ERROR("Cannot add tensors with different shapes!\n");
        return *this;
    }
    Tensor prod(this->shape());
    float * ps = (float *)prod.Tbuf->addr();
    float * p1 = (float *)this->Tbuf->addr();
    float * p2 = (float *)tmul.Tbuf->addr();
    for (int i = 0; i < this->Tsize; i++) {
        ps[i] = p1[i] * p2[i];
    }
    return prod;
}
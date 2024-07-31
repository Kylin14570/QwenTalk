#ifndef NORM_H
#define NORM_H

#include "Macro.h"
#include "Tensor.h"
#include "Loader.h"

class Norm {
    private:
        int Ndim;
        std::shared_ptr<Tensor> Nweight;
        float Neps;
    public:
        Norm(int d, float eps = 0.000001);
        ~Norm() = default;
        void load(Loader * loader, size_t offset);
        Tensor forward(Tensor input);
};

#endif
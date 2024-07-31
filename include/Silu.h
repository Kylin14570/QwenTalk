#ifndef SILU_H
#define SILU_H

#include "Tensor.h"

class Silu {
    public:
        Silu() = default;
        ~Silu() = default;
        static Tensor activate(Tensor input);
};

#endif
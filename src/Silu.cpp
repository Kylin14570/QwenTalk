#include <cmath>
#include "Silu.h"

static inline float silu(float x)
{
    return x / (1.0 + exp(-x));
}

Tensor Silu::activate(Tensor input)
{
    Tensor output = input;
    float * p = (float *)output.host();
    for (int i = 0; i < output.size(); i++)
        p[i] = silu(p[i]);
    return output;
}
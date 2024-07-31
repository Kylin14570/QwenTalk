#include "Norm.h"
#include <cmath>

Norm::Norm(int d, float eps)
{
    Ndim = d;
    Neps = eps;
    Nweight.reset(new Tensor(d));
}

void Norm::load(Loader * loader, size_t offset)
{
    loader->load(offset, Nweight.get());
}

Tensor Norm::forward(Tensor input)
{
    Tensor result(input.shape());
    for (int i = 0; i < input.shape().height; i++) {
        float rms = 0.0f;
        for (int j = 0; j < input.shape().width; j++)
            rms += pow(input.at(i, j), 2.0);
        rms /= (float)input.shape().width;
        rms = sqrt(rms);
        rms += Neps;
        for (int j = 0; j < input.shape().width; j++)
            result.at(i, j) = input.at(i, j) / rms * Nweight->at(j);
    }
    return result;
}
#include "Linear.h"

Linear::Linear(int h, int w, bool hasBias) {
    output_dim = h;
    hidden_dim = w;
    weight.reset(new Tensor(h, w));
    if (hasBias) {
        bias.reset(new Tensor(h));
    } else {
        bias = nullptr;
    }
}

Linear::Linear(const Linear & src)
{
    this->output_dim = src.output_dim;
    this->hidden_dim = src.hidden_dim;
    this->weight.reset(new Tensor(*(src.weight)));
    if (src.bias != nullptr) {
        this->bias.reset(new Tensor(*(src.bias)));
    } else {
        this->bias = nullptr;
    }
}

Linear & Linear::operator= (const Linear & src) {
    if (&src == this) {
        return *this;
    }
    this->output_dim = src.output_dim;
    this->hidden_dim = src.hidden_dim;
    this->weight.reset(new Tensor(*(src.weight)));
    if (src.bias != nullptr) {
        this->bias.reset(new Tensor(*(src.bias)));
    } else {
        this->bias = nullptr;
    }
    return *this;
}

void Linear::load_weight(Loader * loader, size_t offset)
{
    loader->load(offset, weight.get());
}

void Linear::load_bias(Loader * loader, size_t offset)
{
    loader->load(offset, bias.get());
}

Tensor Linear::forward(Tensor input)
{
    int eP = input.shape().height;
    int lP = input.shape().width;
    int hP = this->output_dim;
    Tensor result(eP, hP);
    for (int i = 0; i < eP; i++) {
        for (int j = 0; j < hP; j++) {
            result.at(i, j) = 0;
            for (int k = 0; k < lP; k++) {
                result.at(i, j) += input.at(i, k) * weight->at(j, k);
            }
            if (bias != nullptr) {
                result.at(i, j) += bias->at(i);
            }
        }
    }
    return result;
}
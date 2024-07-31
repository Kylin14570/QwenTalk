#ifndef LINEAR_H
#define LINEAR_H

#include <memory>
#include "Macro.h"
#include "Tensor.h"
#include "Loader.h"

class Linear {
    private:
        int output_dim, hidden_dim;
        std::shared_ptr<Tensor> weight;
        std::shared_ptr<Tensor> bias;
    public:
        Linear(int h, int w, bool hasBias = false);
        Linear(const Linear & src);
        ~Linear() = default;
        Linear & operator= (const Linear & src);
        void load_weight(Loader * loader, size_t offset);
        void load_bias(Loader * loader, size_t offset);
        Tensor forward(Tensor input);
};

#endif
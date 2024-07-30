#ifndef LAYER_H
#define LAYER_H

#include <memory>
#include "Macro.h"
#include "Tensor.h"

class Layer {
    private:
        std::shared_ptr<Tensor> preNorm;
        std::shared_ptr<Tensor> postNorm;
        std::shared_ptr<Tensor> mlpGate;
        std::shared_ptr<Tensor> mlpUp;
        std::shared_ptr<Tensor> mlpDown;
        std::shared_ptr<Tensor> queryWeight;
        std::shared_ptr<Tensor> queryBias;
        std::shared_ptr<Tensor> keyWeight;
        std::shared_ptr<Tensor> keyBias;
        std::shared_ptr<Tensor> valueWeight;
        std::shared_ptr<Tensor> valueBias;
        std::shared_ptr<Tensor> outputLinear;
    public:
        Layer();
        ~Layer();
};

#endif
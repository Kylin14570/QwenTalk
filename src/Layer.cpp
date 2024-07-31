#include "Layer.h"

Layer::Layer()
{
    preNorm.reset(new Tensor(HiddenSize));
    postNorm.reset(new Tensor(HiddenSize));
    mlpGate.reset(new Tensor(InterSize, HiddenSize));
    mlpUp.reset(new Tensor(InterSize, HiddenSize));
    mlpDown.reset(new Tensor(HiddenSize, InterSize));
    queryWeight.reset(new Tensor(NumHead * HeadDim, HiddenSize));
    queryBias.reset(new Tensor(NumHead * HeadDim));
    keyWeight.reset(new Tensor(NumKvHead * HeadDim, HiddenSize));
    keyBias.reset(new Tensor(NumKvHead * HeadDim));
    valueWeight.reset(new Tensor(NumKvHead * HeadDim, HiddenSize));
    valueBias.reset(new Tensor(NumKvHead * HeadDim));
    outputLinear.reset(new Tensor(HiddenSize, HiddenSize));
}
#include <cstdio>
#include <iostream>
#include "Macro.h"
#include "BF16.h"
#include "Tensor.h"
#include "Loader.h"
int main()
{
    printf("Hello\n");
    Tensor embeds(VocabSize, HiddenSize);
    Tensor norm(HiddenSize);
    Loader loader(ModelPath);
    loader.load_embed(&embeds);
    loader.load_norm(&norm);
    printf("Normalization Weight:\n");
    norm.print();
    return 0;
}
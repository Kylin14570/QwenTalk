#include <cstdio>
#include <iostream>
#include <memory>
#include "Macro.h"
#include "BF16.h"
#include "Tensor.h"
#include "Loader.h"
int main()
{
    std::shared_ptr<Loader> loader(new Loader(ModelPath));
    std::shared_ptr<Tensor> embeds(new Tensor(VocabSize, HiddenSize));
    std::shared_ptr<Tensor> norm_weight(new Tensor(HiddenSize));
    printf("Start to load token embeds ...\n");
    size_t bytes = loader->load_embed(embeds.get());
    printf("%lu bytes have been read\n", bytes);
    printf("Done!\n");
    for (int i = 0; i < NumLayer; i++) {
        printf("Start to load layer %d ...\n", i);
        std::shared_ptr<Layer> layer(new Layer);
        bytes = loader->load_layer(layer.get(), i);
        printf("%lu bytes have been read\n", bytes);
        printf("Done!\n");
    }
    printf("Start to load norm weight ...\n");
    bytes = loader->load_norm(norm_weight.get());
    printf("%lu bytes have been read\n", bytes);
    printf("Done!\n");
    printf("Finished successfully!\n");
    return 0;
}
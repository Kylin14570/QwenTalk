#ifndef MACRO_H
#define MACRO_H

#define NumLayer    28
#define NumHead     12
#define NumKvHead   2
#define HeadDim     128
#define HiddenSize  1536
#define InterSize   8960
#define VocabSize   151936
#define Eps         0.000001f

#define ModelPath   "./model.safetensors"

#define HEADER_SIZE 38536UL

#define EMBED_SIZE  (2UL * VocabSize * HiddenSize)

#define LAYER_SIZE  (                                   \
    2UL * (                                             \
        2UL * HiddenSize +                              \
        3UL * HiddenSize * InterSize +                  \
        2UL * NumKvHead * HeadDim * (HiddenSize + 1) +  \
        NumHead * HeadDim * (HiddenSize + 1) +          \
        HiddenSize * HiddenSize                         \
    )                                                   \
)

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#endif
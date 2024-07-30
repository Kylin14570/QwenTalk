#ifndef MACRO_H
#define MACRO_H

#define ModelPath   "./model.safetensors"
#define HEADER_SIZE 38536

#define NumLayer    28
#define NumHead     12
#define NumKvHead   2
#define HeadDim     128
#define HiddenSize  1536
#define InterSize   8960
#define VocabSize   151936
#define EPS         0.000001f

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#endif
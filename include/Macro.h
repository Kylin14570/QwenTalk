#ifndef MACRO_H
#define MACRO_H

#define NumLayer    28UL
#define NumHead     12UL
#define NumKvHead   2UL
#define HeadDim     128UL
#define HiddenSize  1536UL
#define InterSize   8960UL
#define VocabSize   151936UL
#define Eps         0.000001f

#define ModelPath   "../model/model.safetensors"
#define HEADER_SIZE 38536UL

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#define PRINT_ERROR(format, ...) do {                               \
                                    printf("\033[1;31m");           \
                                    printf("ERROR: ");              \
                                    printf(format, ##__VA_ARGS__);  \
                                    printf("\033[0m");              \
                                 } while (false);                   \

#define PRINT_WARNING(format, ...) do {                             \
                                    printf("\033[35m");             \
                                    printf("Warning: ");            \
                                    printf(format, ##__VA_ARGS__);  \
                                    printf("\033[0m");              \
                                 } while (false);                   \
                                 
#endif
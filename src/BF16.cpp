#include "BF16.h"

float bf16_to_float(bf16_t x)
{
    uint32_t dword = (uint32_t)x << 16;
    return *((float *)(&dword));
}

bf16_t float_to_bf16(float x)
{
    uint32_t dword = *((uint32_t *)(&x));
    return (bf16_t)(dword >> 16);
}
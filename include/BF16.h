#ifndef BF16_H
#define BF16_H

#include <stdint.h>
typedef int16_t bf16_t;
float bf16_to_float(bf16_t x);
bf16_t float_to_bf16(float x);

#endif
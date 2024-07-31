#ifndef MALLOC_H
#define MALLOC_H

#include <cstddef>
#include "Macro.h"

void * MemAlloc(size_t size, bool set_zeros = false);

void MemFree(void * addr);

#endif
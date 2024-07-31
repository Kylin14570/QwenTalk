#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "Malloc.h"

void * MemAlloc(size_t size, bool set_zeros)
{
    if (size == 0) {
        PRINT_WARNING("MemAlloc got size = 0\n");
        return NULL;
    }
    void * p = malloc(size);
    if (p == NULL) {
        PRINT_ERROR("MemAlloc Failed!\n");
        exit(0);
    }
    if (set_zeros) {
        memset(p, 0, size);
    }
    return p;
}

void MemFree(void * addr)
{
    if (addr == NULL) {
        PRINT_WARNING("MemFree got addr = NULL\n");
    } else {
        free(addr);
    }
}
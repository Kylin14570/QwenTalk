#ifndef BUFFER_H
#define BUFFER_H

#include <cstddef>
#include "Macro.h"
#include "Malloc.h"

class Buffer {
    private:
        char * mAddr;
        size_t mSize;
    public:
        Buffer();
        Buffer(size_t arg_size, bool set_zeros = false);
        Buffer(const Buffer & src);
        ~Buffer();
        Buffer & operator= (const Buffer & src);
        char * addr();
        size_t size();
};

#endif
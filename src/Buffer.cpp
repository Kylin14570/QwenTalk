#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "Buffer.h"

Buffer::Buffer()
{
    mAddr = NULL;
    mSize = 0;
}

Buffer::Buffer(size_t arg_size, bool set_zeros)
{
    if (arg_size == 0) {
        PRINT_WARNING("A buffer of size 0 is created\n");
        mAddr = NULL;
        mSize = 0;
        return;
    }
    mSize = arg_size;
    mAddr = (char *)MemAlloc(arg_size, set_zeros);
    if (mAddr == NULL) {
        PRINT_ERROR("Failed to create buffer!\n");
        exit(0);
    }
}

Buffer::Buffer(const Buffer & src)
{
    this->mSize = src.mSize;
    this->mAddr = (char *)MemAlloc(this->mSize, false);
    if (this->mAddr == NULL) {
        PRINT_ERROR("Failed to create buffer!\n");
        exit(0);
    }
    memcpy(this->mAddr, src.mAddr, this->mSize);
}

Buffer & Buffer::operator= (const Buffer & src)
{
    if (&src == this) {
        return *this;
    }
    if (this->mAddr) {
        MemFree(this->mAddr);
    }
    this->mSize = src.mSize;
    this->mAddr = (char *)MemAlloc(this->mSize, false);
    if (this->mAddr == NULL) {
        PRINT_ERROR("Failed to create buffer!\n");
        exit(0);
    }
    memcpy(this->mAddr, src.mAddr, this->mSize);
    return *this;
}

Buffer::~Buffer()
{
    if (mAddr) {
        MemFree(mAddr);
    }
}

char * Buffer::addr()
{
    return mAddr;
}

size_t Buffer::size()
{
    return mSize;
}
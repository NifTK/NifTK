#ifndef __CUDA_ALLOC_H__
#define __CUDA_ALLOC_H__

#include "dynlink_cuda.h"
#include <exception>

class CudaAllocator
{
public:
    CudaAllocator(int size_in_bytes) :
        cuda_ptr_(0)
    {
        CUresult r = cuMemAlloc(&cuda_ptr_, size_in_bytes);
        if (r != CUDA_SUCCESS) {
            throw std::runtime_error("Could not allocate memory on the device");
        }
    }

    ~CudaAllocator()
    {
        if (cuda_ptr_) {
            cuMemFree(cuda_ptr_);
        }
    }

    CUdeviceptr * get_pointer()
    {
        return &cuda_ptr_;
    }

private:
    CUdeviceptr					cuda_ptr_;
};

class CudaAutoLock
{
public:
    CudaAutoLock(CUvideoctxlock & lock)
        :lock_(lock)
    {
        cuvidCtxLock(lock_, 0);
    }

    ~CudaAutoLock()
    {
        cuvidCtxUnlock(lock_, 0);
    }

private:
    CUvideoctxlock &			lock_;

};



#endif

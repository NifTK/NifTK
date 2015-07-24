/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef FlipImage_h
#define FlipImage_h

#include "niftkCUDAExports.h"
#include <CUDAManager/CUDAManager.h>


/**
 * Flips the image upside down. This implementation is likely not very efficient.
 * @throws std::runtime_error
 */
void NIFTKCUDA_EXPORT FlipImage(const WriteAccessor& src, WriteAccessor& dest, cudaStream_t stream);

/**
 * Flips the image upside down. This implementation is likely not very efficient.
 * @throws std::runtime_error
 */
void NIFTKCUDA_EXPORT FlipImage(const ReadAccessor& src, WriteAccessor& dest, cudaStream_t stream);


#endif // FlipImage_h

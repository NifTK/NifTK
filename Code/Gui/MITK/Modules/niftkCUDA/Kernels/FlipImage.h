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
 * @throws std::runtime_error
 */
void NIFTKCUDA_EXPORT FlipImage(const WriteAccessor& src, WriteAccessor& dest, cudaStream_t stream);


#endif // FlipImage_h

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkFlipImageLauncher_h
#define niftkFlipImageLauncher_h

#include "niftkCUDAExports.h"
#include <niftkCUDAManager.h>

namespace niftk
{

/**
* \brief Flips the image upside down. This implementation is likely not very efficient.
* \throws std::runtime_error
* TODO: Should not use WriteAccessor, cudaStream in API. Should throw mitk::Exception.
*/
void NIFTKCUDA_EXPORT FlipImageLauncher(const WriteAccessor& src, WriteAccessor& dest, cudaStream_t stream);

/**
* \brief Flips the image upside down. This implementation is likely not very efficient.
* \throws std::runtime_error
* TODO: Should not use ReadAccessor, WriteAccessor, cudaStream in API. Should throw mitk::Exception.
*/
void NIFTKCUDA_EXPORT FlipImageLauncher(const ReadAccessor& src, WriteAccessor& dest, cudaStream_t stream);

} // end namespace

#endif

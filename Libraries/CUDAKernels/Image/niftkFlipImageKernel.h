/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkFlipImageKernel_h
#define niftkFlipImageKernel_h

#include <niftkCUDAKernelsWin32ExportHeader.h>
#include <driver_types.h>

namespace niftk
{

/**
* \brief Flips image in y-axis.
*
* Width and pitch are in bytes.
* Pitch has to be a multiple of 4.
*/
void NIFTKCUDAKERNELS_WINEXPORT RunFlipImageKernel(char* output, int widthInBytes, int height, int outputpitchInBytes, const char* input, int inputpitchInBytes, cudaStream_t stream);

} // end namespace

#endif

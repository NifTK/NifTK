/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef FlipImageKernel_h
#define FlipImageKernel_h

#include <niftkCUDAKernelsWin32ExportHeader.h>
#include <driver_types.h>
//#include <texture_types.h>


/**
 * Width and pitch are in bytes.
 * Pitch has to be a multiple of 4.
 */
void NIFTKCUDAKERNELS_WINEXPORT RunFlipImageKernel(char* output, int widthInBytes, int height, int outputpitchInBytes, const char* input, int inputpitchInBytes, cudaStream_t stream);


#endif // FlipImageKernel_h

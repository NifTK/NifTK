/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef UndistortionKernel_h
#define UndistortionKernel_h

#include <niftkCUDAKernelsWin32ExportHeader.h>
#include <driver_types.h>
#include <texture_types.h>


void NIFTKCUDAKERNELS_WINEXPORT RunUndistortionKernel(char* outputRGBA, int width, int height, cudaTextureObject_t srcTexture, const float* intrinsic3x3, const float* distortion4, cudaStream_t stream);


#endif // UndistortionKernel_h

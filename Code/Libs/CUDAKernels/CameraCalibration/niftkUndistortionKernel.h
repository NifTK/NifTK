/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkUndistortionKernel_h
#define niftkUndistortionKernel_h

#include <niftkCUDAKernelsWin32ExportHeader.h>
#include <driver_types.h>
#include <texture_types.h>

namespace niftk
{

/**
* \brief Provides 2D image undistortion, given a cameras intrinsic and distortion co-efficients.
*/
void NIFTKCUDAKERNELS_WINEXPORT RunUndistortionKernel(char* outputRGBA, int width, int height, cudaTextureObject_t srcTexture, const float* intrinsic3x3, const float* distortion4, cudaStream_t stream);

} // end namespace

#endif

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkEdgeDetectionKernel_h
#define niftkEdgeDetectionKernel_h

#include <niftkCUDAKernelsWin32ExportHeader.h>
#include <driver_types.h>
#include <texture_types.h>

namespace niftk
{

/**
* \brief Basic example kernel to do some edge detection, but is not yet implemented.
*/
void NIFTKCUDAKERNELS_WINEXPORT RunEdgeDetectionKernel(char* outputRGBA, unsigned int outputBytePitch, const char* inputRGBA, unsigned int inputBytePitch, int width, int height, cudaStream_t stream);

} // end namespace

#endif

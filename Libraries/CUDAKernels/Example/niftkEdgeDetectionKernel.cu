/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkEdgeDetectionKernel.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sm_30_intrinsics.h>
#include <common_functions.h>
#include <texture_types.h>
#include <driver_types.h>

#ifdef WIN32
#include <cassert>
#endif

namespace niftk
{

//-----------------------------------------------------------------------------
__global__ void edgedetection_kernel(char* outputRGBA, unsigned int outputPixelPitch, const char* inputRGBA, unsigned int inputPixelPitch, int width, int height)
{
#ifdef WIN32
  // should be static assert
  assert(sizeof(uchar4) == sizeof(unsigned int));
#endif

  // these are output coordinates.
  int   x = blockIdx.x * blockDim.x + threadIdx.x;
  int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // if the input image to our kernel has an odd size then x and y can be out of bounds.
  // this is because we round up the launch config to multiples of 32 or 16.
  if ((x < width) && (y < height))
  {
    uchar4*        outRGBA = &(((      uchar4*) outputRGBA)[y * outputPixelPitch + x]);
    const uchar4*  inRGBA  = &(((const uchar4*) inputRGBA )[y * inputPixelPitch  + x]);

    *outRGBA = *inRGBA;
  }
}


//-----------------------------------------------------------------------------
void RunEdgeDetectionKernel(char* outputRGBA, unsigned int outputBytePitch, const char* inputRGBA, unsigned int inputBytePitch, int width, int height, cudaStream_t stream)
{
#ifdef WIN32
  // should be static assert
  assert(sizeof(uchar4) == sizeof(unsigned int));

  assert((outputBytePitch % 4) == 0);
  assert((inputBytePitch % 4) == 0);
  assert((width * 4) <= outputBytePitch);
  assert((width * 4) <= inputBytePitch);
#endif

  // launch config
  dim3  threads(32, 16);
  dim3  grid((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

  // note to self: the third param is "dynamically allocated shared mem".
  edgedetection_kernel<<<grid, threads, 0, stream>>>(outputRGBA, outputBytePitch / 4, inputRGBA, inputBytePitch / 4, width, height);
}

} // end namespace

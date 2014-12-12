/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "../niftkCUDAKernelsWin32ExportHeader.h"
#include "EdgeDetectionKernel.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sm_30_intrinsics.h>
#include <common_functions.h>
#include <texture_types.h>
#include <driver_types.h>


//-----------------------------------------------------------------------------
__global__ void edgedetection_kernel(char* outputRGBA, int width, int height, cudaTextureObject_t texture)
{
  // these are output coordinates.
  int   x = blockIdx.x * blockDim.x + threadIdx.x;
  int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // if the input image to our kernel has an odd size then x and y can be out of bounds.
  // this is because we round up the launch config to multiples of 32 or 16.
  if ((x < width) && (y < height))
  {
    unsigned int*  outRGBA = &(((unsigned int*) outputRGBA)[y * width + x]);

    float4  pixel = tex2D<float4>(texture, x, y);

    unsigned int    out =
        ((unsigned int) (pixel.x * 255))
     | (((unsigned int) (pixel.y * 255)) << 8)
     | (((unsigned int) (pixel.z * 255)) << 16)
     | (((unsigned int) (pixel.w * 255)) << 24);

    *outRGBA = out;
  }
}


//-----------------------------------------------------------------------------
void RunEdgeDetectionKernel(char* outputRGBA, int width, int height, cudaTextureObject_t srcTexture, cudaStream_t stream)
{
  // launch config
  dim3  threads(32, 16);
  dim3  grid((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

  // note to self: the third param is "dynamically allocated shared mem".
  edgedetection_kernel<<<grid, threads, 0, stream>>>(outputRGBA, width, height, srcTexture);
}

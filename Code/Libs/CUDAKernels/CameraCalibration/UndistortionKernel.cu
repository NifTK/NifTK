/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "../niftkCUDAKernelsWin32ExportHeader.h"
#include "UndistortionKernel.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <sm_30_intrinsics.h>
#include <cassert>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <common_functions.h>


//-----------------------------------------------------------------------------
__global__ void donothing_kernel()
{

}


//-----------------------------------------------------------------------------
__global__ void undistortion_kernel(char* outputRGBA, int width, int height, cudaTextureObject_t texture, float intrinsic[3*3], float distortion[4])
{
  // these are output coordinates.
  int   x = blockIdx.x * blockDim.x + threadIdx.x;
  int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // if the input image to our kernel has an odd size then x and y can be out of bounds.
  // this because we round up the launch config to multiples of 32 or 16.
  if ((x < width) && (y < height))
  {
    unsigned int*  outRGBA = &(((unsigned int*) outputRGBA)[y * width + x]);

    // map output backwards through the camera & distortion to find out where we need to read from.
    float   px = ((float) x + 0.5f) / (float) width;
    float   py = ((float) y + 0.5f) / (float) height;

    float4  pixel = tex2D<float4>(texture, px, py);

    unsigned int    out =
        ((unsigned int) (pixel.x * 255))
     | (((unsigned int) (pixel.y * 255)) << 8)
     | (((unsigned int) (pixel.z * 255)) << 16)
     | (((unsigned int) (pixel.w * 255)) << 24);

    *outRGBA = out;
  }
}


//-----------------------------------------------------------------------------
void NIFTKCUDAKERNELS_WINEXPORT RunUndistortionKernel(char* outputRGBA, int width, int height, cudaTextureObject_t texture, const float* intrinsic3x3, const float* distortion4, cudaStream_t stream)
{
  float   intrinsic[3*3];
  std::memcpy(&intrinsic[0], intrinsic3x3, sizeof(intrinsic));

  float   distortion[4];
  std::memcpy(&distortion[0], distortion4, sizeof(distortion));

  // launch config
  dim3  threads(32, 16);
  dim3  grid((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

  // note to self: the third param is "dynamically allocated shared mem".
  undistortion_kernel<<<grid, threads, 0, stream>>>(outputRGBA, width, height, texture, intrinsic, distortion);
}

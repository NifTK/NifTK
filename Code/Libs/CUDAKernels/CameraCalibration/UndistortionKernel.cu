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
#include <cstring>
#include <common_functions.h>


//-----------------------------------------------------------------------------
__global__ void donothing_kernel()
{

}


template <typename T, int S>
struct Array
{
  T     data[S];
};


//-----------------------------------------------------------------------------
__global__ void undistortion_kernel(char* outputRGBA, int width, int height, cudaTextureObject_t texture, Array<float, 3 * 3> intrinsic, Array<float, 4> distortion)
{
  // these are output coordinates.
  int   x = blockIdx.x * blockDim.x + threadIdx.x;
  int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // if the input image to our kernel has an odd size then x and y can be out of bounds.
  // this because we round up the launch config to multiples of 32 or 16.
  if ((x < width) && (y < height))
  {
    unsigned int*  outRGBA = &(((unsigned int*) outputRGBA)[y * width + x]);

    // normalise coordinate with respect to (de-)centered pinhole.
    float   nx = (x - intrinsic.data[2]) / intrinsic.data[0];
    float   ny = (y - intrinsic.data[5]) / intrinsic.data[4];

    // input to this is the undistorted coordinate (our output coordinate)
    // and we want to find out where it would be distorted to.

    // radial distortion.
    float   r  = (nx * nx) + (ny * ny);
    float   dx = nx * (1.0f + distortion.data[0] * r + distortion.data[1] * r * r);
    float   dy = ny * (1.0f + distortion.data[0] * r + distortion.data[1] * r * r);
    // tangential distortion.
    dx += distortion.data[3] * (r + 2.0f * nx * nx) + (2.0f * distortion.data[2] * nx * ny);
    dy += distortion.data[2] * (r + 2.0f * ny * ny) + (2.0f * distortion.data[3] * nx * ny);

    // undo coordinate normalisation (back to pixel coordinates).
    float   ux = dx * intrinsic.data[0] + intrinsic.data[2];
    float   uy = dy * intrinsic.data[4] + intrinsic.data[5];

    // adjust for opengl 0.5 pixel offset: sample at center of the pixel.
    // and normalise to texture coordinates.
    ux = (ux + 0.5f) / (float) width;
    uy = (uy + 0.5f) / (float) height;

    float4  pixel = tex2D<float4>(texture, ux, uy);

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
  Array<float, 3 * 3>   intrinsic;
  std::memcpy(&intrinsic.data[0], intrinsic3x3, sizeof(intrinsic));

  Array<float, 4>       distortion;
  std::memcpy(&distortion.data[0], distortion4, sizeof(distortion));

  // launch config
  dim3  threads(32, 16);
  dim3  grid((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

  // note to self: the third param is "dynamically allocated shared mem".
  undistortion_kernel<<<grid, threads, 0, stream>>>(outputRGBA, width, height, texture, intrinsic, distortion);
}

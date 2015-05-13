/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "FlipImageKernel.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <cassert>


//-----------------------------------------------------------------------------
__global__ void flipimage_kernel(unsigned int* output, int width, int height, int outputpitch, unsigned int* input, int inputpitch)
{
  // these are output coordinates, in units of 4-byte.
  int   x = blockIdx.x * blockDim.x + threadIdx.x;
  int   y = blockIdx.y * blockDim.y + threadIdx.y;

  // if the input image to our kernel has an odd size then x and y can be out of bounds.
  // this because we round up the launch config to powers of two.
  if ((x < width) && (y < height))
  {
    unsigned int*  out = &output[outputpitch * y + x];

    // input is flipped.
    int     inx = x;
    int     iny = height - y - 1;

    assert(iny >= 0);
    assert(iny < height);
    assert(inx >= 0);
    assert(inx < width);

    unsigned int*   in = &input[inputpitch * iny + inx];

    *out = *in;
  }
}


//-----------------------------------------------------------------------------
void RunFlipImageKernel(char* output, int widthInBytes, int height, int outputpitchInBytes, const char* input, int inputpitchInBytes, cudaStream_t stream)
{
  if ((outputpitchInBytes % 4) != 0)
    throw std::runtime_error("Pitch has to be a multiple of 4");
  if ((inputpitchInBytes % 4) != 0)
    throw std::runtime_error("Pitch has to be a multiple of 4");

  if (widthInBytes > outputpitchInBytes)
    throw std::runtime_error("Width is larger than pitch");
  if (widthInBytes > inputpitchInBytes)
    throw std::runtime_error("Width is larger than pitch");

  int   width = (widthInBytes + 3) / 4;


  // launch config
  dim3  threads(128, 8);
  dim3  grid((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

  // note to self: the third param is "dynamically allocated shared mem".
  flipimage_kernel<<<grid, threads, 0, stream>>>((unsigned int*) output, width, height, outputpitchInBytes / 4, (unsigned int*) input, inputpitchInBytes / 4);
}

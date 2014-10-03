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
void NIFTKCUDAKERNELS_WINEXPORT RunDoNothingKernel(cudaStream_t stream)
{
  cudaError_t err = cudaSuccess;

  dim3  grid(4, 4);
  dim3  threads(32, 4);

  // note to self: the third param is "dynamically allocated shared mem".
  donothing_kernel<<<grid, threads, 0, stream>>>();
}

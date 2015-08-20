/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkCUDAUtils_h
#define niftkCUDAUtils_h

#include <niftkCUDAKernelsWin32ExportHeader.h>
#include <cuda.h>
#include <cstdio>

namespace niftk
{

/**
* \file niftkCUDAUtils.h
* \brief Various simple utilities for CUDA work.
*/

/**
* \brief Static method (only available to same T.U.) to print error.
*/
static void HandleCUDAError( cudaError_t err,
                             const char *file,
                             int line ) {
  if (err != cudaSuccess) {
    printf( "HandleCUDAError: Error=%i, message=%s in %s at line %d\n", err, cudaGetErrorString( err ), file, line );
    exit( EXIT_FAILURE );
  }
}

/**
* \brief Macro to check and print error code.
*/
#define niftkCUDACall( err ) ( HandleCUDAError( err, __FILE__, __LINE__ ))

} // end namespace

#endif

/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2010-09-14 15:39:17 +0100 (Tue, 14 Sep 2010) $
 Revision          : $Revision: 3898 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef CUDA_UTILS_GPU_CU
#define CUDA_UTILS_GPU_CU

#include <stdio.h>
#include "cuda_utils_kernels.cu"
#include "cuda_utils_gpu.h"

/**
 * In this file we define/implement the C functions defined in cuda_utils_gpu.h
 * Each function can call any C functions, and call many GPU kernels, taking care
 * of whether memory is created/used/destroyed on host or on device.
 */
 
/** Test function to add two floats. */
float TestAdd(float a, float b)
{
  float c;
  float *dev_c;
  
  cudaMalloc((void**)&dev_c, sizeof(float));
  
  TestAddKernel<<<1,1>>>(a, b, dev_c);
  
  cudaMemcpy(&c, dev_c, sizeof(float), cudaMemcpyDeviceToHost);  
  cudaFree(dev_c);
  
  return c;
}

#endif CUDA_UTILS_GPU_CU
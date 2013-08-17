/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <itkLogHelper.h>
#include <niftkConversionUtils.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_utils_gpu.h>
#include <string.h>
#include <stdio.h>

/*!
 * \file niftkCUDAInfo.cxx
 * \page niftkCUDAInfo
 * \section niftkCUDAInfoSummary Prints out information on the GPU enabled devices on the current machine.
 */
void Usage(char *exec)
  {
    niftk::itkLogHelper::PrintCommandLineHeader(std::cout);
    std::cout << "  " << std::endl;
    std::cout << "  Prints out information on the GPU enabled devices on the current machine." << std::endl;
    std::cout << "  Test program copied from 'deviceQuery' that ships with NVidia SDK" << std::endl;
    std::cout << "  " << std::endl;
    std::cout << "  Useage: " << exec << std::endl;
    std::cout << "  " << std::endl;
  }

/**
 * \brief Prints out CUDA capabilities of the installed cards
 */
int main(int argc, char** argv)
{
  float a, b;
  bool gotA = false;
  bool gotB = false;
  

  // Parse command line args
  for(int i=1; i < argc; i++){
    if(strcmp(argv[i], "-help")==0 || strcmp(argv[i], "-Help")==0 || strcmp(argv[i], "-HELP")==0 || strcmp(argv[i], "-h")==0 || strcmp(argv[i], "--h")==0){
      Usage(argv[0]);
      return -1;
    }
    else if(strcmp(argv[i], "-a") == 0){
      a=atof(argv[++i]);
      gotA = true;
      std::cout << "Set -a=" << niftk::ConvertToString(a) << std::endl;
    }
    else if(strcmp(argv[i], "-b") == 0){
      b=atof(argv[++i]);
      gotB = true;
      std::cout << "Set -b=" << niftk::ConvertToString(b) << std::endl;
    }
    else {
      std::cerr << argv[0] << ":\tParameter " << argv[i] << " unknown." << std::endl;
      return -1;
    }
  }
  if (gotA && gotB)
  {
    float c = TestAdd(a,b);
    std::cerr << a << " + " << b << " = " << c << std::endl;
  }

  unsigned long int tempULongInt;
  int dev;
  int deviceCount;
  int driverVersion = 0, runtimeVersion = 0;

  cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
  if (cudaResultCode != cudaSuccess)
  {
    deviceCount = 0;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0)
  {
    printf("There are no devices supporting CUDA.\n");
    return EXIT_FAILURE;
  }

  for (dev = 0; dev < deviceCount; ++dev) {

      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, dev);

      if (dev == 0) {
          // This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
          if (deviceProp.major == 9999 && deviceProp.minor == 9999)
          {
            printf("There is no device supporting CUDA.\n");
            return EXIT_FAILURE;
          }
          else if (deviceCount == 1)
          {
            printf("There is 1 device supporting CUDA\n");
          }
          else
          {
            printf("There are %d devices supporting CUDA\n", deviceCount);
          }
      }
      printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

#if CUDART_VERSION >= 2020
      cudaDriverGetVersion(&driverVersion);
      printf("  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
      cudaRuntimeGetVersion(&runtimeVersion);
      printf("  CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
#endif
      printf("  CUDA Capability Major revision number:         %d\n", deviceProp.major);
      printf("  CUDA Capability Minor revision number:         %d\n", deviceProp.minor);
      tempULongInt = deviceProp.totalGlobalMem;
      printf("  Total amount of global memory:                 %lu bytes\n", tempULongInt);
#if CUDART_VERSION >= 2000
      printf("  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
#endif
      tempULongInt = deviceProp.totalConstMem;
      printf("  Total amount of constant memory:               %lu bytes\n", tempULongInt);
      tempULongInt = deviceProp.sharedMemPerBlock;
      printf("  Total amount of shared memory per block:       %lu bytes\n", tempULongInt);
      printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
      printf("  Warp size:                                     %d\n", deviceProp.warpSize);
      printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
      printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
             deviceProp.maxThreadsDim[0],
             deviceProp.maxThreadsDim[1],
             deviceProp.maxThreadsDim[2]);
      printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
             deviceProp.maxGridSize[0],
             deviceProp.maxGridSize[1],
             deviceProp.maxGridSize[2]);
      tempULongInt = deviceProp.memPitch;
      printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
      tempULongInt = deviceProp.textureAlignment;
      printf("  Texture alignment:                             %lu bytes\n", tempULongInt);
      printf("  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
#if CUDART_VERSION >= 2000
      printf("  Concurrent copy and execution:                 %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
#endif
#if CUDART_VERSION >= 2020
      printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
      printf("  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
      printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
      printf("  Compute mode:                                  %s\n", deviceProp.computeMode == cudaComputeModeDefault ?
                                                                "Default (multiple host threads can use this device simultaneously)" :
                                                                  deviceProp.computeMode == cudaComputeModeExclusive ?
                                  "Exclusive (only one host thread at a time can use this device)" :
                                                                  deviceProp.computeMode == cudaComputeModeProhibited ?
                                  "Prohibited (no host thread can use this device)" :
                                  "Unknown");
  #endif
  #if CUDART_VERSION >= 3000
      printf("  Concurrent kernel execution:                   %s\n", deviceProp.concurrentKernels ? "Yes" : "No");
  #endif
  #if CUDART_VERSION >= 3010
      printf("  Device has ECC support enabled:                %s\n", deviceProp.ECCEnabled ? "Yes" : "No");
  #endif
  } // end for loop
  return EXIT_SUCCESS;
}

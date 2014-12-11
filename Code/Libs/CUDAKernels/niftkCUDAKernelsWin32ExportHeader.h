/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkCUDAKernelsWin32ExportHeader_h
#define niftkCUDAKernelsWin32ExportHeader_h

#include <NifTKConfigure.h>

#if (defined(_WIN32) || defined(WIN32)) && !defined(NIFTK_STATIC) 
  #ifdef niftkCUDAKernels_EXPORTS
    #define NIFTKCUDAKERNELS_WINEXPORT __declspec(dllexport)
  #else
    #define NIFTKCUDAKERNELS_WINEXPORT __declspec(dllimport)
  #endif
#else
/* linux/mac needs nothing */
  #define NIFTKCUDAKERNELS_WINEXPORT
#endif


#endif  //niftkCUDAKernelsWin32ExportHeader_h

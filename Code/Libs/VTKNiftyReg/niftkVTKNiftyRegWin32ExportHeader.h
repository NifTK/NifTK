/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkVTKNiftyRegWin32ExportHeader_h
#define niftkVTKNiftyRegWin32ExportHeader_h

#include <NifTKConfigure.h>

#if (defined(_WIN32) || defined(WIN32)) && !defined(NIFTK_STATIC) 
  #ifdef NIFTKVTKNIFTYREGWINDOWS_EXPORT
    #define NIFTKVTKNIFTYREG_WINEXPORT __declspec(dllexport)
  #else
    #define NIFTKVTKNIFTYREG_WINEXPORT __declspec(dllimport)
  #endif  /* NIFTKVTKNIFTYREG_WINEXPORT */
#else
/* linux/mac needs nothing */
  #define NIFTKVTKNIFTYREG_WINEXPORT
#endif


#endif  //niftkVTKNiftyRegWin32ExportHeader_h

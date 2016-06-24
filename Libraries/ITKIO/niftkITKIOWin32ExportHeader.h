/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkITKIOWin32ExportHeader_h
#define niftkITKIOWin32ExportHeader_h

#include <NifTKConfigure.h>

#if (defined(_WIN32) || defined(WIN32)) && !defined(NIFTK_STATIC) 
  #ifdef NIFTKITKIO_WINDOWS_EXPORT
    #define NIFTKITKIO_WINEXPORT __declspec(dllexport)
  #else
    #define NIFTKITKIO_WINEXPORT __declspec(dllimport)
  #endif  /* NIFTKITKIO_WINEXPORT */
#else
/* linux/mac needs nothing */
  #define NIFTKITKIO_WINEXPORT 
#endif


#endif  //__NIFTKITKWIN32EXPORTHEADER_H

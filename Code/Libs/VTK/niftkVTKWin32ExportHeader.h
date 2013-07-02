/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __NIFTKVTKWIN32EXPORTHEADER_H
#define __NIFTKVTKWIN32EXPORTHEADER_H

#include <NifTKConfigure.h>

#if (defined(_WIN32) || defined(WIN32)) && !defined(NIFTK_STATIC) 
  #ifdef NIFTKVTK_WINDOWS_EXPORT
    #define NIFTKVTK_WINEXPORT __declspec(dllexport)
  #else
    #define NIFTKVTK_WINEXPORT __declspec(dllimport)
  #endif  /* NIFTKVTK_WINEXPORT */
#else
/* linux/mac needs nothing */
  #define NIFTKVTK_WINEXPORT 
#endif


#endif  //__NIFTKVTKWIN32EXPORTHEADER_H

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __NIFTKMIDASWIN32EXPORTHEADER_H
#define __NIFTKMIDASWIN32EXPORTHEADER_H

#include "NifTKConfigure.h"

#if (defined(_WIN32) || defined(WIN32)) && !defined(NIFTK_STATIC) 
  #ifdef NIFTKMIDAS_WINDOWS_EXPORT
    #define NIFTKMIDAS_WINEXPORT __declspec(dllexport)
  #else
    #define NIFTKMIDAS_WINEXPORT __declspec(dllimport)
  #endif  /* NIFTKMIDAS_WINEXPORT */
#else
/* linux/mac needs nothing */
  #define NIFTKMIDAS_WINEXPORT 
#endif


#endif  //__NIFTKMIDASWIN32EXPORTHEADER_H

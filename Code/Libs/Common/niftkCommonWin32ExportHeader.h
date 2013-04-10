/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __NIFTKCOMMONWIN32EXPORTHEADER_H
#define __NIFTKCOMMONWIN32EXPORTHEADER_H

#include "NifTKConfigure.h"

#if (defined(_WIN32) || defined(WIN32)) && !defined(NIFTK_STATIC) 
  #ifdef NIFTKCOMMON_WINDOWS_EXPORT
    #define NIFTKCOMMON_WINEXPORT __declspec(dllexport)
  #else
    #define NIFTKCOMMON_WINEXPORT __declspec(dllimport)
  #endif  /* NIFTKCOMMON_WINEXPORT */
#else
/* linux/mac needs nothing */
  #define NIFTKCOMMON_WINEXPORT 
#endif


#endif  //__NIFTKCOMMONWIN32EXPORTHEADER_H

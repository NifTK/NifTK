/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __NIFTKITKWIN32EXPORTHEADER_H
#define __NIFTKITKWIN32EXPORTHEADER_H

#include "NifTKConfigure.h"

#if (defined(_WIN32) || defined(WIN32)) && !defined(NIFTK_STATIC) 
  #ifdef NIFTKITK_WINDOWS_EXPORT
    #define NIFTKITK_WINEXPORT __declspec(dllexport)
  #else
    #define NIFTKITK_WINEXPORT __declspec(dllimport)
  #endif  /* NIFTKITK_WINEXPORT */
#else
/* linux/mac needs nothing */
  #define NIFTKITK_WINEXPORT 
#endif


#endif  //__NIFTKITKWIN32EXPORTHEADER_H

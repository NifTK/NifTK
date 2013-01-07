/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.
 
 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-07 16:28:33 +0100 (Thu, 07 Jul 2011) $
 Revision          : $Revision: 6690 $
 Last modified by  : $Author: ad $

 Original author   : a.duttaroy@cs.ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __NIFTKNVAPIWIN32EXPORTHEADER_H
#define __NIFTKNVAPIWIN32EXPORTHEADER_H

#include "NifTKConfigure.h"

#if (defined(_WIN32) || defined(WIN32)) && !defined(NIFTK_STATIC) 
  #ifdef NIFTKNVAPI_WINDOWS_EXPORT
    #define NIFTKNVAPI_WINEXPORT __declspec(dllexport)
  #else
    #define NIFTKNVAPI_WINEXPORT __declspec(dllimport)
  #endif  /* NIFTKNVAPI_WINEXPORT */
#else
/* linux/mac needs nothing */
  #define NIFTKNVAPI_WINEXPORT
#endif


#endif  //__NIFTKNVIDIAWIN32EXPORTHEADER_H

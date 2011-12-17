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
#ifndef __NIFTKQTWIN32EXPORTHEADER_H
#define __NIFTKQTWIN32EXPORTHEADER_H

#include "NifTKConfigure.h"
#include <QtCore/QtGlobal>

#if (defined(_WIN32) || defined(WIN32)) && !defined(NIFTK_STATIC) 
  #ifdef NIFTKQT_WINDOWS_EXPORT
    #define NIFTKQT_WINEXPORT Q_DECL_EXPORT
  #else
    #define NIFTKQT_WINEXPORT Q_DECL_IMPORT
  #endif  /* NIFTKQT_WINEXPORT */
#else
/* linux/mac needs nothing */
  #define NIFTKQT_WINEXPORT 
#endif


#endif  //__NIFTKQTWIN32EXPORTHEADER_H

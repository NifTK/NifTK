/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-01 19:03:07 +0100 (Fri, 01 Jul 2011) $
 Revision          : $Revision: 6628 $
 Last modified by  : $Author: ad $

 Original author   : j.hipwell@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#ifndef __itkFilenameUtils_h
#define __itkFilenameUtils_h

#include "NifTKConfigure.h"
#include "niftkITKWin32ExportHeader.h"

#include <string>
#include "itkObject.h"

namespace itk
{

  /** Modifies the suffix (taking into account .Z or .gz) */
  extern "C++" NIFTKITK_WINEXPORT   ITK_EXPORT std::string ModifyFilenameSuffix( std::string filename, std::string suffix );

  /** Extract and return the suffix (taking into account .Z or .gz) */
  extern "C++" NIFTKITK_WINEXPORT   ITK_EXPORT std::string ExtractSuffix( std::string filename, std::string suffix );

} // end namespace

#endif

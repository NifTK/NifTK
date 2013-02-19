/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef __itkFilenameUtils_h
#define __itkFilenameUtils_h

#include "NifTKConfigure.h"
#include "niftkITKWin32ExportHeader.h"

#include <string>
#include "itkObject.h"

namespace itk
{

  /** Modifies the suffix (taking into account .Z or .gz) */
  extern "C++" NIFTKITK_WINEXPORT ITK_EXPORT std::string ModifyFilenameSuffix( std::string filename, std::string suffix );

  /** Extract and return the suffix (taking into account .Z or .gz) */
  extern "C++" NIFTKITK_WINEXPORT ITK_EXPORT std::string ExtractSuffix( std::string filename, std::string suffix );

} // end namespace

#endif

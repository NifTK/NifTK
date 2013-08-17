/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef itkConversionUtils_h
#define itkConversionUtils_h

#include <NifTKConfigure.h>
#include <niftkITKWin32ExportHeader.h>

#include <string>
#include <itkObject.h>
#include <itkSpatialOrientation.h>

namespace itk
{

  /** Converts a SpatialOrientation code into a string. eg. ITK_COORDINATE_ORIENTATION_RAS to RAS. */
  extern "C++" NIFTKITK_WINEXPORT ITK_EXPORT std::string ConvertSpatialOrientationToString(const SpatialOrientation::ValidCoordinateOrientationFlags &code);

  /** Converts a code (3 letters, uppercase, made from L/R, I/S, A/P), into a ValidCoordinateOrientationFlags. */
  extern "C++" NIFTKITK_WINEXPORT ITK_EXPORT SpatialOrientation::ValidCoordinateOrientationFlags ConvertStringToSpatialOrientation(std::string code);

} // end namespace
#endif

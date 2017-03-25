/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMITKMathsUtils_h
#define niftkMITKMathsUtils_h

#include "niftkCoreExports.h"
#include <mitkPoint.h>
#include <mitkVector.h>
#include <vtkMatrix4x4.h>

/**
 * \file niftkMITKMathsUtils.h
 * \brief Various math stuff that also uses MITK data-types.
 */
namespace niftk {

/**
* \brief Converts a rotation quaternion and translation vector to a 4x4 rigid body matrix.
*/
NIFTKCORE_EXPORT void ConvertRotationAndTranslationToMatrix(const mitk::Point4D& rotation,
                                                            const mitk::Vector3D& translation,
                                                            vtkMatrix4x4& matrix
                                                           );


/**
* \brief Converts a 4x4 rigid body matrix to a rotation quaternion and translation vector.
*/
NIFTKCORE_EXPORT void ConvertMatrixToRotationAndTranslation(const vtkMatrix4x4& matrix,
                                                            mitk::Point4D& rotation,
                                                            mitk::Vector3D& translation
                                                           );
}

#endif

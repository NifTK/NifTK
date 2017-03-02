/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMITKMathsUtils.h"
#include <vtkMath.h>

namespace niftk
{

//-----------------------------------------------------------------------------
void ConvertRotationAndTranslationToMatrix(const mitk::Point4D& rotation,
                                           const mitk::Vector3D& translation,
                                           vtkMatrix4x4& matrix
                                          )
{
  double quaternion[4];
  quaternion[0] = rotation[0];
  quaternion[1] = rotation[1];
  quaternion[2] = rotation[2];
  quaternion[3] = rotation[3];

  double rotationMatrix[3][3];
  vtkMath::QuaternionToMatrix3x3(quaternion, rotationMatrix);

  matrix.Identity();
  for (int r = 0; r < 3; r++)
  {
    for (int c = 0; c < 3; c++)
    {
      matrix.SetElement(r, c, rotationMatrix[r][c]);
    }
    matrix.SetElement(r, 3, translation[r]);
  }
}


//-----------------------------------------------------------------------------
void ConvertMatrixToRotationAndTranslation(const vtkMatrix4x4& matrix,
                                           mitk::Point4D& rotation,
                                           mitk::Vector3D& translation
                                           )
{
  translation[0] = matrix.GetElement(0, 3);
  translation[1] = matrix.GetElement(1, 3);
  translation[2] = matrix.GetElement(2, 3);

  double rotationMatrix[3][3]
    = {
        { matrix.GetElement(0, 0), matrix.GetElement(0, 1), matrix.GetElement(0, 2) },
        { matrix.GetElement(1, 0), matrix.GetElement(1, 1), matrix.GetElement(1, 2) },
        { matrix.GetElement(2, 0), matrix.GetElement(2, 1), matrix.GetElement(2, 2) },
      };

  double quaternion[4];
  vtkMath::Matrix3x3ToQuaternion(rotationMatrix, quaternion);

  rotation[0] = quaternion[0];
  rotation[1] = quaternion[1];
  rotation[2] = quaternion[2];
  rotation[3] = quaternion[3];
}

} // end namespace

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkLiuLeastSquaresWithNormalsRegistrationWrapper.h"
#include "mitkLiuLeastSquaresWithNormalsRegistration.h"
#include <cv.h>
#include <mitkOpenCVMaths.h>

namespace mitk {

//-----------------------------------------------------------------------------
LiuLeastSquaresWithNormalsRegistrationWrapper::LiuLeastSquaresWithNormalsRegistrationWrapper()
{
}


//-----------------------------------------------------------------------------
LiuLeastSquaresWithNormalsRegistrationWrapper::~LiuLeastSquaresWithNormalsRegistrationWrapper()
{
}


//-----------------------------------------------------------------------------
bool LiuLeastSquaresWithNormalsRegistrationWrapper::Update(const mitk::PointSet::Pointer& fixedPoints,
                                                           const mitk::PointSet::Pointer& fixedNormals,
                                                           const mitk::PointSet::Pointer& movingPoints,
                                                           const mitk::PointSet::Pointer& movingNormals,
                                                           vtkMatrix4x4& matrix,
                                                           double &fiducialRegistrationError)
{
  std::vector<cv::Point3d> fP = mitk::PointSetToVector(fixedPoints);
  std::vector<cv::Point3d> fN = mitk::PointSetToVector(fixedNormals);
  std::vector<cv::Point3d> mP = mitk::PointSetToVector(movingPoints);
  std::vector<cv::Point3d> mN = mitk::PointSetToVector(movingNormals);
  cv::Matx44d registrationMatrix;

  mitk::LiuLeastSquaresWithNormalsRegistration::Pointer registration = mitk::LiuLeastSquaresWithNormalsRegistration::New();
  bool result = registration->Update(fP, fN, mP, mN, registrationMatrix, fiducialRegistrationError);

  CopyToVTK4x4Matrix(registrationMatrix, matrix);

  return result;

}

//-----------------------------------------------------------------------------
} // end namespace

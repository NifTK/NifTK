/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkArunLeastSquaresPointRegistrationWrapper.h"
#include "mitkArunLeastSquaresPointRegistration.h"
#include <cv.h>
#include <mitkOpenCVMaths.h>

namespace mitk
{

//-----------------------------------------------------------------------------
ArunLeastSquaresPointRegistrationWrapper::ArunLeastSquaresPointRegistrationWrapper()
{
}


//-----------------------------------------------------------------------------
ArunLeastSquaresPointRegistrationWrapper::~ArunLeastSquaresPointRegistrationWrapper()
{
}


//-----------------------------------------------------------------------------
bool ArunLeastSquaresPointRegistrationWrapper::Update(const mitk::PointSet::Pointer& fixedPoints,
                                                      const mitk::PointSet::Pointer& movingPoints,
                                                      vtkMatrix4x4& matrix,
                                                      double& fiducialRegistrationError
                                                     )
{
  std::vector<cv::Point3d> fP = PointSetToVector(fixedPoints);
  std::vector<cv::Point3d> mP = PointSetToVector(movingPoints);
  cv::Matx44d registrationMatrix;

  mitk::ArunLeastSquaresPointRegistration::Pointer registration = mitk::ArunLeastSquaresPointRegistration::New();
  bool result = registration->Update(fP, mP, registrationMatrix, fiducialRegistrationError);

  CopyToVTK4x4Matrix(registrationMatrix, matrix);

  return result;
}

//-----------------------------------------------------------------------------
} // end namespace

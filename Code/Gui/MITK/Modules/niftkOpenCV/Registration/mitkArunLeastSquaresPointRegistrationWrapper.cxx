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
// Private method. Don't make this part of public API. ToDo: PIMPL pattern.
//-----------------------------------------------------------------------------
std::vector<cv::Point3d> PointSetToVector(const mitk::PointSet::Pointer& pointSet)
{
  std::vector<cv::Point3d> result;

  // This assumes that the pointID's come out ordered?
  // ToDo: Check if that is the case, and if not, sort by ID.

  mitk::PointSet::DataType* itkPointSet = pointSet->GetPointSet(0);
  mitk::PointSet::PointsContainer* points = itkPointSet->GetPoints();
  mitk::PointSet::PointsIterator pIt;
  mitk::PointSet::PointType point;

  for (pIt = points->Begin(); pIt != points->End(); ++pIt)
  {
    point = pIt->Value();
    cv::Point3d cvPoint;

    cvPoint.x = point[0];
    cvPoint.y = point[1];
    cvPoint.z = point[2];
    result.push_back(cvPoint);
  }

  return result;
}


//-----------------------------------------------------------------------------
bool ArunLeastSquaresPointRegistrationWrapper::Update(const mitk::PointSet::Pointer& fixedPoints,
                                                      const mitk::PointSet::Pointer& movingPoints,
                                                      vtkMatrix4x4& matrix,
                                                      double& fiducialRegistrationError
                                                     )
{
  std::vector<cv::Point3d> fixed = PointSetToVector(fixedPoints);
  std::vector<cv::Point3d> moving = PointSetToVector(movingPoints);
  cv::Matx44d registrationMatrix;

  mitk::ArunLeastSquaresPointRegistration::Pointer registration = mitk::ArunLeastSquaresPointRegistration::New();
  bool result = registration->Update(fixed, moving, registrationMatrix, fiducialRegistrationError);

  for (unsigned int i = 0; i < 4; ++i)
  {
    for (unsigned int j = 0; j < 4; j++)
    {
      matrix.SetElement(i, j, registrationMatrix(i,j));
    }
  }

  return result;
}

//-----------------------------------------------------------------------------
} // end namespace

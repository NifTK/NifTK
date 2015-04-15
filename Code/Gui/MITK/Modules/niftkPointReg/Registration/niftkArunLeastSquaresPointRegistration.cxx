/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkArunLeastSquaresPointRegistration.h"
#include <mitkOpenCVMaths.h>
#include <mitkExceptionMacro.h>
#include <niftkPointRegMaths.h>

namespace niftk {

//-----------------------------------------------------------------------------
double PointBasedRegistration(const std::vector<cv::Point3d>& fixedPoints,
                              const std::vector<cv::Point3d>& movingPoints,
                              cv::Matx44d& outputMatrix
                             )
{
  if (fixedPoints.size() < 3)
  {
    mitkThrow() << "The number of 'fixed' points is < 3";
  }

  if (movingPoints.size() < 3)
  {
    mitkThrow() << "The number of 'moving' points is < 3";
  }

  if (fixedPoints.size() != movingPoints.size())
  {
    mitkThrow() << "The number of 'fixed' points is " << fixedPoints.size()
                << " whereas the number of 'moving' points is " << movingPoints.size()
                << " and they should correspond.";
  }

  // Arun Equation 4.
  cv::Point3d pPrime = mitk::GetCentroid(fixedPoints);

  // Arun Equation 6.
  cv::Point3d p = mitk::GetCentroid(movingPoints);

  // Arun Equation 7.
  std::vector<cv::Point3d> q = mitk::SubtractPointFromPoints(movingPoints, p);

  // Arun Equation 8.
  std::vector<cv::Point3d> qPrime = mitk::SubtractPointFromPoints(fixedPoints, pPrime);

  // Arun Equation 11.
  cv::Matx33d H = niftk::CalculateCrossCovarianceH(q, qPrime);

  // Delegate to helper method for the rest of the implementation.
  double fre = niftk::DoSVDPointBasedRegistration(fixedPoints, movingPoints, H, p, pPrime, outputMatrix);
  return fre;
}


//-----------------------------------------------------------------------------
double PointBasedRegistration(const mitk::PointSet::Pointer& fixedPoints,
                              const mitk::PointSet::Pointer& movingPoints,
                              vtkMatrix4x4& matrix)
{
  std::vector<cv::Point3d> fP = mitk::PointSetToVector(fixedPoints);
  std::vector<cv::Point3d> mP = mitk::PointSetToVector(movingPoints);

  cv::Matx44d mat;
  double fre = niftk::PointBasedRegistration(fP, mP, mat);
  mitk::CopyToVTK4x4Matrix(mat, matrix);

  return fre;
}

//-----------------------------------------------------------------------------
} // end namespace






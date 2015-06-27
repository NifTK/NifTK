/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkLiuLeastSquaresWithNormalsRegistration.h"
#include <niftkPointRegMaths.h>
#include <mitkOpenCVMaths.h>
#include <mitkExceptionMacro.h>

namespace niftk {

//-----------------------------------------------------------------------------
double PointAndNormalBasedRegistrationUsingSVD(const std::vector<cv::Point3d>& fixedPoints,
                                               const std::vector<cv::Point3d>& fixedNormals,
                                               const std::vector<cv::Point3d>& movingPoints,
                                               const std::vector<cv::Point3d>& movingNormals,
                                               cv::Matx44d& outputMatrix)
{
  if (fixedPoints.size() < 2)
  {
    mitkThrow() << "The number of 'fixed' points is < 2";
  }

  if (fixedNormals.size() < 2)
  {
    mitkThrow() << "The number of 'fixed' normals is < 2";
  }

  if (movingPoints.size() < 2)
  {
    mitkThrow() << "The number of 'moving' points is < 2";
  }

  if (movingNormals.size() < 2)
  {
    mitkThrow() << "The number of 'moving' normals is < 2";
  }

  if (fixedPoints.size() != movingPoints.size())
  {
    mitkThrow() << "The number of 'fixed' points is " << fixedPoints.size()
                << " whereas the number of 'moving' points is " << movingPoints.size()
                << " and they should correspond.";
  }

  if (fixedNormals.size() != movingNormals.size())
  {
    mitkThrow() << "The number of 'fixed' normals is " << fixedNormals.size()
                << " whereas the number of 'moving' normals is " << movingNormals.size()
                << " and they should correspond.";
  }

  if (fixedPoints.size() != fixedNormals.size())
  {
    mitkThrow() << "The number of 'fixed' points is " << fixedPoints.size()
                << " whereas the number of 'fixed' normals is " << fixedNormals.size()
                << " and they should correspond.";
  }

  // See: http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=758228

  // Liu Equation 12.
  cv::Point3d pBar = mitk::GetCentroid(movingPoints);
  cv::Point3d qBar = mitk::GetCentroid(fixedPoints);

  // Liu Equation 14.
  std::vector<cv::Point3d> pTilde = mitk::SubtractPointFromPoints(movingPoints, pBar);
  std::vector<cv::Point3d> qTilde = mitk::SubtractPointFromPoints(fixedPoints, qBar);

  // Liu Equation 13.
  cv::Matx33d H1 = niftk::CalculateCrossCovarianceH(pTilde, qTilde);
  cv::Matx33d H2 = niftk::CalculateCrossCovarianceH(movingNormals, fixedNormals);
  cv::Matx33d H = H1 + H2;

  // Delegate to helper method for the rest of the implementation.
  double fre = niftk::DoSVDPointBasedRegistration(fixedPoints, movingPoints, H, qBar, qBar, outputMatrix);
  return fre;
}


//-----------------------------------------------------------------------------
double PointAndNormalBasedRegistrationUsingSVD(const mitk::PointSet::Pointer fixedPoints,
                                               const mitk::PointSet::Pointer fixedNormals,
                                               const mitk::PointSet::Pointer movingPoints,
                                               const mitk::PointSet::Pointer movingNormals,
                                               vtkMatrix4x4& matrix)
{
  if (fixedPoints.IsNull())
  {
    mitkThrow() << "The 'fixed' points are NULL";
  }
  if (fixedNormals.IsNull())
  {
    mitkThrow() << "The 'fixed' normals are NULL";
  }
  if (movingPoints.IsNull())
  {
    mitkThrow() << "The 'moving' points are NULL";
  }
  if (movingNormals.IsNull())
  {
    mitkThrow() << "The 'moving' normals are NULL";
  }

  std::vector<cv::Point3d> fP = mitk::PointSetToVector(fixedPoints);
  std::vector<cv::Point3d> fN = mitk::PointSetToVector(fixedNormals);
  std::vector<cv::Point3d> mP = mitk::PointSetToVector(movingPoints);
  std::vector<cv::Point3d> mN = mitk::PointSetToVector(movingNormals);

  cv::Matx44d mat;
  double fre = niftk::PointAndNormalBasedRegistrationUsingSVD(fP, fN, mP, mN, mat);
  mitk::CopyToVTK4x4Matrix(mat, matrix);

  return fre;
}

//-----------------------------------------------------------------------------
} // end namespace

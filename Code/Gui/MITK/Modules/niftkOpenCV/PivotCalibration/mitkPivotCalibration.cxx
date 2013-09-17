/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkPivotCalibration.h"
#include <mitkFileIOUtils.h>
#include <niftkFileHelper.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <mitkCameraCalibrationFacade.h>
#include <mitkOpenCVMaths.h>

namespace mitk {

//-----------------------------------------------------------------------------
PivotCalibration::PivotCalibration()
  : m_SingularValueThreshold(0.01)
{
}


//-----------------------------------------------------------------------------
PivotCalibration::~PivotCalibration()
{
}


//-----------------------------------------------------------------------------
bool PivotCalibration::CalibrateUsingFilesInDirectories(
    const std::string& matrixDirectory,
    double &residualError,
    vtkMatrix4x4& outputMatrix
    )
{
  std::vector<cv::Mat> matrices = LoadMatricesFromDirectory (matrixDirectory);
  if (matrices.size() == 0)
  {
    MITK_ERROR << "Calibrate: Failed to load matrices." << std::endl;
    return false;
  }

  cv::Matx44d transformationMatrix;

  bool calibratedSuccessfully = this->Calibrate(
      matrices,
      transformationMatrix,
      residualError
      );

  if (!calibratedSuccessfully)
  {
    MITK_ERROR << "Calibrate: Failed to calibrate successfully" << std::endl;
    return false;
  }

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      outputMatrix.SetElement(i, j, transformationMatrix(i, j));
    }
  }
  return true;
}


//-----------------------------------------------------------------------------
bool PivotCalibration::Calibrate(
    const std::vector< cv::Mat >& matrices,
    cv::Matx44d& outputMatrix,
    double& residualError
    )
{
  bool isSuccessful = false;

  unsigned long int numberOfMatrices = matrices.size();

  // Very basic. Will eventually run out of memory.

  cv::Mat a = cvCreateMat(3*numberOfMatrices, 6,CV_64FC1);
  cv::Mat x = cvCreateMat(6, 1,CV_64FC1);
  cv::Mat b = cvCreateMat(3*numberOfMatrices, 1,CV_64FC1);

  // Fill matrices.
  for (unsigned long int i = 0; i < numberOfMatrices; i++)
  {
    b.at<double>(i * 3 + 0, 0) = -1 * matrices[i].at<double>(0, 3);
    b.at<double>(i * 3 + 1, 0) = -1 * matrices[i].at<double>(1, 3);
    b.at<double>(i * 3 + 2, 0) = -1 * matrices[i].at<double>(2, 3);

    a.at<double>(i * 3 + 0, 0) = matrices[i].at<double>(0, 0);
    a.at<double>(i * 3 + 1, 0) = matrices[i].at<double>(1, 0);
    a.at<double>(i * 3 + 2, 0) = matrices[i].at<double>(2, 0);
    a.at<double>(i * 3 + 0, 1) = matrices[i].at<double>(0, 1);
    a.at<double>(i * 3 + 1, 1) = matrices[i].at<double>(1, 1);
    a.at<double>(i * 3 + 2, 1) = matrices[i].at<double>(2, 1);
    a.at<double>(i * 3 + 0, 2) = matrices[i].at<double>(0, 2);
    a.at<double>(i * 3 + 1, 2) = matrices[i].at<double>(1, 2);
    a.at<double>(i * 3 + 2, 2) = matrices[i].at<double>(2, 2);

    a.at<double>(i * 3 + 0, 3) = -1;
    a.at<double>(i * 3 + 1, 3) =  0;
    a.at<double>(i * 3 + 2, 3) =  0;
    a.at<double>(i * 3 + 0, 4) =  0;
    a.at<double>(i * 3 + 1, 4) = -1;
    a.at<double>(i * 3 + 2, 4) =  0;
    a.at<double>(i * 3 + 0, 5) =  0;
    a.at<double>(i * 3 + 1, 5) =  0;
    a.at<double>(i * 3 + 2, 5) = -1;

  }

  cv::SVD svdOfA(a);

  // Zero out diagonal values less than threshold
  int rank = 0;
  for (long int i = 0; i < svdOfA.w.rows; i++)
  {
    if (svdOfA.w.at<double>(i, 0) < m_SingularValueThreshold)
    {
      svdOfA.w.at<double>(i, 0) = 0;
    }

    if (svdOfA.w.at<double>(i, 0) != 0)
    {
      rank++;
    }
  }

  if (rank < 6)
  {
    std::cerr << "PivotCalibration: Failed. Rank < 6" << std::endl;
    return isSuccessful;
  }

  svdOfA.backSubst(b, x);

  // Calculate residual.
  cv::Mat residualMatrix = (a*x - b);
  residualError = 0;
  for (unsigned long int i = 0; i < numberOfMatrices*3; i++)
  {
    residualError += residualMatrix.at<double>(i, 0)*residualMatrix.at<double>(i, 0);
  }
  residualError /= (double)(numberOfMatrices*3);
  residualError = sqrt(residualError);

  // Prepare output
  MakeIdentity(outputMatrix);
  outputMatrix(0, 3) = x.at<double>(0, 0);
  outputMatrix(1, 3) = x.at<double>(1, 0);
  outputMatrix(2, 3) = x.at<double>(2, 0);
  isSuccessful = true;

  std::cout << "PivotCalibration:Residual error   = " << residualError << std::endl;
  std::cout << "PivotCalibration:Pivot            = (" << x.at<double>(3, 0) << ", " << x.at<double>(4, 0) << ", " << x.at<double>(5, 0) << ")" << std::endl;
  std::cout << "PivotCalibration:Result:" << std::endl;
  for (int i = 0; i < 4; i++)
  {
    std::cout << outputMatrix(i, 0) << " " << outputMatrix(i, 1) << " " << outputMatrix(i, 2) << " " << outputMatrix(i, 3) << std::endl;
  }
  return isSuccessful;
}

//-----------------------------------------------------------------------------
} // end namespace

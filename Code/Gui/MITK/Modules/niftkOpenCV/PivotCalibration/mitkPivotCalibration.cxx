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
#include <stdexcept>
#include <sstream>
#include <cstdlib>

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
void PivotCalibration::CalibrateUsingFilesInDirectories(
    const std::string& matrixDirectory,
    double &residualError,
    vtkMatrix4x4& outputMatrix,
    const int& percentage,
    const int& reruns
    )
{
  std::vector<cv::Mat> matrices = LoadMatricesFromDirectory (matrixDirectory);
  if (matrices.size() == 0)
  {
    std::ostringstream oss;
    oss << "Calibrate: Failed to load matrices." << std::endl;
    throw std::logic_error(oss.str());
  }

  cv::Matx44d transformationMatrix;
  
  this->Calibrate(
    matrices,
    transformationMatrix,
    residualError,
    percentage,
    reruns
    );

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      outputMatrix.SetElement(i, j, transformationMatrix(i, j));
    }
  }
}


//-----------------------------------------------------------------------------
void PivotCalibration::Calibrate(
    const std::vector< cv::Mat >& matrices,
    cv::Matx44d& outputMatrix,
    double& residualError,
    const int& percentage,
    const int& numberOfReRuns
    )
{
  // So the main output is always ALL the data.
  this->DoCalibration(matrices, outputMatrix, residualError);
  
  if (percentage < 100 && percentage > 0)
  {
    unsigned int totalNumberOfMatrices = matrices.size();
    unsigned int numberOfMatricesToUse = totalNumberOfMatrices * percentage / 100;
    if (numberOfMatricesToUse < 1)
    {
      std::cerr << "PivotCalibration: too few matrices to calculate mean (StdDev) of residual error" << std::endl;
      return;
    }
    
    std::cout << "PivotCalibration:Total #matrices = " << totalNumberOfMatrices << std::endl;
    std::cout << "PivotCalibration:Using " << percentage << "%" << std::endl;
    std::cout << "PivotCalibration:Rerunning " << numberOfReRuns << " times" << std::endl;
    
    std::vector<double> vectorOfResiduals;
    std::vector<double> xOffset;
    std::vector<double> yOffset;
    std::vector<double> zOffset;
    double tmpResidual = 0;

    cv::Matx44d tmpMatrix;
    
    for (unsigned int i = 0; i < numberOfReRuns; i++)
    {
      // Build randomly chosen set of indexes of size numberOfMatricesToUse.
      // TODO: std::rand() is known to be non-uniform in this scenario at least.
      //       this will preferentially pick lower numbers.
      std::set<int> matrixIndexes;
      while(matrixIndexes.size() < numberOfMatricesToUse)
      {
        matrixIndexes.insert(std::rand()%totalNumberOfMatrices);
      }
      
      // Construct vector of matrices from set.
      std::vector< cv::Mat > randomlyChosenMatrices;
      std::set<int>::const_iterator iter;
      for (iter = matrixIndexes.begin(); iter != matrixIndexes.end(); ++iter)
      {
        randomlyChosenMatrices.push_back(matrices[*iter]);    
      }
      
      // Calibrate using this list of matrices, and store the residual.
      this->DoCalibration(randomlyChosenMatrices, tmpMatrix, tmpResidual);
      vectorOfResiduals.push_back(tmpResidual);
      xOffset.push_back(tmpMatrix(0,3));
      yOffset.push_back(tmpMatrix(1,3));
      zOffset.push_back(tmpMatrix(2,3));
    }
    double meanResidual = mitk::Mean(vectorOfResiduals);
    double stdDevResidual = mitk::StdDev(vectorOfResiduals);
    double meanX = mitk::Mean(xOffset);
    double meanY = mitk::Mean(yOffset);
    double meanZ = mitk::Mean(zOffset);
    double stdDevX = mitk::StdDev(xOffset);
    double stdDevY = mitk::StdDev(yOffset);
    double stdDevZ = mitk::StdDev(zOffset);

    std::cout << "PivotCalibration:Rerunning " << numberOfReRuns << " times on " << percentage << "% of the data gives offset=[" \
              << meanX << " (" << stdDevX << "), " \
              << meanY << " (" << stdDevY << "), " \
              << meanZ << " (" << stdDevZ << ")] " \
              << "residual=" << meanResidual << " ( " << stdDevResidual << ")" << std::endl;
  }
}


//-----------------------------------------------------------------------------
void PivotCalibration::DoCalibration(
    const std::vector< cv::Mat >& matrices,
    cv::Matx44d& outputMatrix,
    double& residualError
    )
{
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
    std::ostringstream oss;
    oss << "PivotCalibration: Failed. Rank < 6" << std::endl;
    throw std::logic_error(oss.str());
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

  std::cout << "PivotCalibration:Pivot = (" << x.at<double>(3, 0) << ", " << x.at<double>(4, 0) << ", " << x.at<double>(5, 0) << "), residual=" << residualError << std::endl;
}

//-----------------------------------------------------------------------------
} // end namespace

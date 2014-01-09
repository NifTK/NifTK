/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <limits>
#include <mitkUltrasoundPinCalibration.h>
#include <niftkUltrasoundPinCalibrationCLP.h>
#include <mitkVector.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>
#include <niftkVTKFunctions.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;
  double residualError = std::numeric_limits<double>::max();

  if ( matrixDirectory.length() == 0 || pointDirectory.length() == 0)
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    mitk::Point3D invPoint;
    if (invariantPoint.size() == 3)
    {
      invPoint[0] = invariantPoint[0];
      invPoint[1] = invariantPoint[1];
      invPoint[2] = invariantPoint[2];
    }
    else
    {
      invPoint[0] = 0;
      invPoint[1] = 0;
      invPoint[2] = 0;
    }

    vtkSmartPointer<vtkMatrix4x4> initialMatrix = vtkMatrix4x4::New();
    initialMatrix->Identity();

    if(initialGuess.size() != 0)
    {
      initialMatrix = niftk::LoadMatrix4x4FromFile(initialGuess, false);
    }
    cv::Matx33d rotationMatrix;
    cv::Matx31d rotationVector;
    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        rotationMatrix(i,j) = initialMatrix->GetElement(i,j);
      }
    }
    cv::Rodrigues(rotationMatrix, rotationVector);

    std::vector<double> initialTransformationParameters;

    initialTransformationParameters.push_back(rotationVector(0,0));
    initialTransformationParameters.push_back(rotationVector(1,0));
    initialTransformationParameters.push_back(rotationVector(2,0));
    initialTransformationParameters.push_back(initialMatrix->GetElement(0,3));
    initialTransformationParameters.push_back(initialMatrix->GetElement(1,3));
    initialTransformationParameters.push_back(initialMatrix->GetElement(2,3));

    if (optimiseScaling)
    {
      initialTransformationParameters.push_back(millimetresPerPixel);
    }

    if (optimiseInvariantPoint)
    {
      initialTransformationParameters.push_back(invPoint[0]);
      initialTransformationParameters.push_back(invPoint[1]);
      initialTransformationParameters.push_back(invPoint[2]);
    }

    std::cout << "niftkUltrasoundPinCalibration: matrices         = " << matrixDirectory << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: points           = " << pointDirectory << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: output           = " << outputMatrixFile << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: invariant point  = " << invPoint << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: mm/pix           = " << millimetresPerPixel << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: optimising       = ";
    for (unsigned int i = 0; i < initialTransformationParameters.size(); i++)
    {
      std::cout << initialTransformationParameters[i] << " ";
    }
    std::cout << std::endl;

    vtkSmartPointer<vtkMatrix4x4> transformationMatrix = vtkMatrix4x4::New();
    double mmPerPix = millimetresPerPixel;

    // Do calibration
    mitk::UltrasoundPinCalibration::Pointer calibration = mitk::UltrasoundPinCalibration::New();
    bool isSuccessful = calibration->CalibrateUsingInvariantPointAndFilesInTwoDirectories(
        matrixDirectory,
        pointDirectory,
        optimiseScaling,
        optimiseInvariantPoint,
        initialTransformationParameters,
        invPoint,
        mmPerPix,
        residualError,
        *transformationMatrix
        );

    std::cout << "niftkUltrasoundPinCalibration: residual         = " << residualError << std::endl;

    if (isSuccessful)
    {
      if (niftk::SaveMatrix4x4ToFile(outputMatrixFile, *transformationMatrix))
      {
        returnStatus = EXIT_SUCCESS;
      }
      else
      {
        MITK_ERROR << "Failed to save transformation to file:" << outputMatrixFile << std::endl;
        returnStatus = EXIT_FAILURE;
      }
    }
    else
    {
      MITK_ERROR << "Calibration failed" << std::endl;
      returnStatus = EXIT_FAILURE;
    }
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception:" << e.what();
    returnStatus = -1;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:";
    returnStatus = -2;
  }

  return returnStatus;
}

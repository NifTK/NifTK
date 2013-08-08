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

    mitk::Point2D originInPixels;
    if (pixelOrigin.size() == 2)
    {
      originInPixels[0] = pixelOrigin[0];
      originInPixels[1] = pixelOrigin[1];
    }
    else
    {
      originInPixels[0] = 0;
      originInPixels[1] = 0;
    }

    mitk::Point2D mmPerPix;
    if (millimetresPerPixel.size() == 2)
    {
      mmPerPix[0] = millimetresPerPixel[0];
      mmPerPix[1] = millimetresPerPixel[1];
    }
    else
    {
      mmPerPix[0] = 1;
      mmPerPix[1] = 1;
    }

    vtkSmartPointer<vtkMatrix4x4> trackerToPhantomMatrix = vtkMatrix4x4::New();
    trackerToPhantomMatrix->Identity();
    if (trackerToPhantomMatrixFile.size() > 0)
    {
      trackerToPhantomMatrix = niftk::LoadMatrix4x4FromFile(trackerToPhantomMatrixFile);
    }

    std::vector<double> initialTransformationParameters;
    if(initialGuess.size() == 6)
    {
      initialTransformationParameters.push_back(initialGuess[0]);
      initialTransformationParameters.push_back(initialGuess[1]);
      initialTransformationParameters.push_back(initialGuess[2]);
      initialTransformationParameters.push_back(initialGuess[3]);
      initialTransformationParameters.push_back(initialGuess[4]);
      initialTransformationParameters.push_back(initialGuess[5]);
    }
    else
    {
      initialTransformationParameters.push_back(0);
      initialTransformationParameters.push_back(0);
      initialTransformationParameters.push_back(0);
      initialTransformationParameters.push_back(0);
      initialTransformationParameters.push_back(0);
      initialTransformationParameters.push_back(0);
    }
    if (optimisingScaling)
    {
      initialTransformationParameters.push_back(mmPerPix[0]);
      initialTransformationParameters.push_back(mmPerPix[1]);
    }

    std::cout << "niftkUltrasoundPinCalibration: matrices         = " << matrixDirectory << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: points           = " << pointDirectory << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: output           = " << outputMatrixFile << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: invariant point  = " << invPoint << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: origin in pixels = " << originInPixels << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: mm/pix           = " << mmPerPix << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: optimising       = ";
    for (unsigned int i = 0; i < initialTransformationParameters.size(); i++)
    {
      std::cout << initialTransformationParameters[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: tracker-phantom  = " << std::endl;
    for (int i = 0; i < 4; i++)
    {
      std::cout << "niftkUltrasoundPinCalibration:   " << trackerToPhantomMatrix->GetElement(i,0) << " " << trackerToPhantomMatrix->GetElement(i,1) << " " << trackerToPhantomMatrix->GetElement(i,2) << " " << trackerToPhantomMatrix->GetElement(i,3) << std::endl;
    }

    // Do calibration
    mitk::UltrasoundPinCalibration::Pointer calibration = mitk::UltrasoundPinCalibration::New();
    bool isSuccessful = calibration->CalibrateUsingInvariantPointAndFilesInTwoDirectories(
        matrixDirectory,
        pointDirectory,
        *trackerToPhantomMatrix,
        invPoint,
        originInPixels,
        mmPerPix,
        initialTransformationParameters,
        residualError,
        outputMatrixFile
        );

    if (isSuccessful)
    {
      std::cout << "Residual error=" << residualError << std::endl;
      returnStatus = EXIT_SUCCESS;
    }
    else
    {
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

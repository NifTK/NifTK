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
    mitk::Point3D invariantPoint;
    invariantPoint[0] = 0;
    invariantPoint[1] = 0;
    invariantPoint[2] = 0;

    mitk::Point2D originInPixels;
    originInPixels[0] = xOrigin;
    originInPixels[1] = yOrigin;

    vtkSmartPointer<vtkMatrix4x4> trackerToPhantomMatrix = vtkMatrix4x4::New();
    trackerToPhantomMatrix->Identity();
    if (trackerToPhantomMatrixFile.size() > 0)
    {
      trackerToPhantomMatrix = niftk::LoadMatrix4x4FromFile(trackerToPhantomMatrixFile);
    }

    mitk::Point2D millimetresPerPixel;
    millimetresPerPixel[0] = 1;
    millimetresPerPixel[1] = 1;

    std::vector<double> initialGuessTransformation;
    initialGuessTransformation.push_back(0); // rx
    initialGuessTransformation.push_back(0); // ry
    initialGuessTransformation.push_back(0); // rz
    initialGuessTransformation.push_back(0); // tx
    initialGuessTransformation.push_back(0); // ty
    initialGuessTransformation.push_back(0); // tz

    // Do calibration
    mitk::UltrasoundPinCalibration::Pointer calibration = mitk::UltrasoundPinCalibration::New();
    bool isSuccessful = calibration->CalibrateUsingInvariantPointAndFilesInTwoDirectories(
        matrixDirectory,
        pointDirectory,
        *trackerToPhantomMatrix,
        invariantPoint,
        originInPixels,
        millimetresPerPixel,
        initialGuessTransformation,
        residualError,
        outputMatrix
        );

    if (isSuccessful)
    {
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

  std::cout << "Residual error=" << residualError << ", return status = " << returnStatus << std::endl;
  return returnStatus;
}

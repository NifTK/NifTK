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
    originInPixels[0] = 0;
    originInPixels[1] = 0;

    mitk::UltrasoundPinCalibration::Pointer calibration = mitk::UltrasoundPinCalibration::New();

    bool isSuccessful = calibration->CalibrateUsingTrackerPointAndFilesInTwoDirectories(
        matrixDirectory,
        pointDirectory,
        outputMatrix,
        invariantPoint,
        originInPixels,
        residualError
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

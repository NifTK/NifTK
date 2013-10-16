/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <QmitkIGIUtils.h>

#include <niftkIGIMakeGeometryCLP.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if ( leftCameraInputDirectory.length() == 0 || outputCalibrationData.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    if (   (leftCameraInputDirectory.length() != 0 && rightCameraInputDirectory.length()  == 0)
        || (rightCameraInputDirectory.length() != 0 && leftCameraInputDirectory.length() == 0)
        )
    {
      mitk::CameraCalibrationFromDirectory::Pointer calibrationObject = mitk::CameraCalibrationFromDirectory::New();
      if (rightCameraInputDirectory.length() != 0)
      {
        reprojectionError = calibrationObject->Calibrate(rightCameraInputDirectory, xCorners, yCorners, size, outputCalibrationData, writeImages);
      }
      else
      {
        reprojectionError = calibrationObject->Calibrate(leftCameraInputDirectory, xCorners, yCorners, size, outputCalibrationData, writeImages);
      }
    }
    else
    {
      mitk::StereoCameraCalibrationFromTwoDirectories::Pointer calibrationObject = mitk::StereoCameraCalibrationFromTwoDirectories::New();
      reprojectionError = calibrationObject->Calibrate(leftCameraInputDirectory, rightCameraInputDirectory, xCorners, yCorners, size, outputCalibrationData, writeImages);
    }
    returnStatus = EXIT_SUCCESS;
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

  std::cout << "Reprojection error=" << reprojectionError << ", return status = " << returnStatus << std::endl;
  return returnStatus;
}

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
#include <stdexcept>
#include <mitkCameraCalibrationFromDirectory.h>
#include <mitkStereoCameraCalibration.h>
#include <niftkCameraCalibrationCLP.h>
#include <mitkVector.h>
#include <niftkFileHelper.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;
  double reprojectionError = std::numeric_limits<double>::max();

  if ( leftCameraInputDirectory.length() == 0 || outputDirectory.length() == 0 )
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  if (numberOfFrames != 0 && leftCameraInputDirectory.length() != 0 && rightCameraInputDirectory.length()  != 0)
  {
    MITK_ERROR << "If you specify --numberFrames, you must only specify --left OR --right, and not both" << std::endl;
    return returnStatus;
  }

  try
  {
    mitk::Point2D pixelScales;
    pixelScales[0] = pixelScaleFactors[0];
    pixelScales[1] = pixelScaleFactors[1];

    if (!niftk::DirectoryExists(outputDirectory))
    {
      if (!niftk::CreateDirAndParents(outputDirectory))
      {
        MITK_ERROR << "Failed to create directory:" << outputDirectory << std::endl;
        return -3;
      }
    }
    
    if (numberOfFrames == 0
        && ((leftCameraInputDirectory.length() != 0 && rightCameraInputDirectory.length()  == 0)
            || (rightCameraInputDirectory.length() != 0 && leftCameraInputDirectory.length() == 0)
           )
        )
    {
      mitk::CameraCalibrationFromDirectory::Pointer calibrationObject = mitk::CameraCalibrationFromDirectory::New();
      if (rightCameraInputDirectory.length() != 0)
      {
        reprojectionError = calibrationObject->Calibrate(rightCameraInputDirectory, xCorners, yCorners, size, pixelScales, outputDirectory, writeImages);
      }
      else
      {
        reprojectionError = calibrationObject->Calibrate(leftCameraInputDirectory, xCorners, yCorners, size, pixelScales, outputDirectory, writeImages);
      }
    }
    else
    {
      mitk::StereoCameraCalibration::Pointer calibrationObject = mitk::StereoCameraCalibration::New();

      if ( existingCalibrationDirectory != "" )
      {
        MITK_INFO << "Attempting to use existing intrinsic calibration from " << existingCalibrationDirectory;
        calibrationObject->LoadExistingIntrinsics(existingCalibrationDirectory);
      }

      reprojectionError = calibrationObject->Calibrate(leftCameraInputDirectory, rightCameraInputDirectory, numberOfFrames, xCorners, yCorners, size, pixelScales, outputDirectory, writeImages);
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

  std::cout << "Reprojection error = " << reprojectionError << ", return status = " << returnStatus << std::endl;
  return returnStatus;
}

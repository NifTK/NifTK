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
#include <mitkVector.h>

#include <mitkCameraCalibrationFromDirectory.h>
#include <mitkStereoCameraCalibration.h>
#include <niftkFileHelper.h>
#include <niftkCameraCalibrationCLP.h>

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

      if ( existingIntrinsicsDirectory != "" )
      {
        MITK_INFO << "Attempting to use existing intrinsic calibration from " << existingIntrinsicsDirectory;
        calibrationObject->LoadExistingIntrinsics(existingIntrinsicsDirectory);
      }

      if ( existingRightToLeftDirectory != "" )
      {
        MITK_INFO << "Attempting to use existing right-to-left calibration from " << existingRightToLeftDirectory;
        calibrationObject->LoadExistingRightToLeft(existingRightToLeftDirectory);
      }

      reprojectionError = calibrationObject->Calibrate(leftCameraInputDirectory, rightCameraInputDirectory, numberOfFrames, xCorners, yCorners, size, pixelScales, outputDirectory, writeImages);
    }
    returnStatus = EXIT_SUCCESS;
  }
  catch (mitk::Exception& e)
  {
    MITK_ERROR << "Caught mitk::Exception: " << e.GetDescription() << ", from:" << e.GetFile() << "::" << e.GetLine() << std::endl;
    returnStatus = EXIT_FAILURE + 100;
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception: " << e.what() << std::endl;
    returnStatus = EXIT_FAILURE + 101;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:" << std::endl;
    returnStatus = EXIT_FAILURE + 102;
  }

  std::cout << "Reprojection error = " << reprojectionError << ", return status = " << returnStatus << std::endl;
  return returnStatus;
}

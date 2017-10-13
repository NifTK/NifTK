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

#include <mitkCameraCalibrationFacade.h>
#include <niftkHandeyeCalibrateUsingRegistration.h>
#include <niftkHandeyeCalibrationUsingRegistrationCLP.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if ( modelInputFile.length() == 0 && modelTrackingDirectory.length() == 0)
  {
    std::cerr << "Error: no model data specified!" << std::endl;
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  if ( cameraPointsDirectory.length() == 0)
  {
    std::cerr << "Error: no camera data specified!" << std::endl;
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    niftk::HandeyeCalibrateUsingRegistration::Pointer calibrator = niftk::HandeyeCalibrateUsingRegistration::New();
    calibrator->Calibrate(
      modelInputFile,
      modelTrackingDirectory,
      cameraPointsDirectory,
      handTrackingDirectory,
      distanceThreshold,
      errorThreshold,
      output
      );
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
  return returnStatus;
}

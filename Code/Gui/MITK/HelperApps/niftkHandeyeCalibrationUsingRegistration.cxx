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
#include <mitkHandeyeCalibrateUsingRegistration.h>
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
    mitk::HandeyeCalibrateUsingRegistration::Pointer calibrator = mitk::HandeyeCalibrateUsingRegistration::New();
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

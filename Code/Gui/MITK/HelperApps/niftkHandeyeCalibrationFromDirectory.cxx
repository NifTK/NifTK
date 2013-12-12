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
#include <mitkHandeyeCalibrateFromDirectory.h>
#include <niftkHandeyeCalibrationFromDirectoryCLP.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  bool sortByDistance = !DontSortByDistance;
  try
  {
    mitk::Point2D pixelScales;
    pixelScales[0] = pixelScaleFactors[0];
    pixelScales[1] = pixelScaleFactors[1];

    mitk::HandeyeCalibrateFromDirectory::Pointer calibrator = mitk::HandeyeCalibrateFromDirectory::New();
    calibrator->SetInputDirectory(trackingInputDirectory);
    calibrator->SetOutputDirectory(outputDirectory);
    calibrator->SetTrackerIndex(trackerIndex);
    calibrator->SetAbsTrackerTimingError(MaxTimingError);
    calibrator->SetFramesToUse(FramesToUse);
    calibrator->SetSortByDistance(sortByDistance);
    calibrator->SetFlipTracking(FlipTracking);
    calibrator->SetFlipExtrinsic(FlipExtrinsic);
    calibrator->SetSortByAngle(false);
    calibrator->SetPixelScaleFactor(pixelScales);
    calibrator->SetSwapVideoChannels(swapVideoChannels);
    calibrator->InitialiseTracking();

    if ( existingCalibrationDirectory != "" ) 
    {
      MITK_INFO << "Attempting to use existing intrinsic calibration from " << existingCalibrationDirectory;
      calibrator->LoadExistingIntrinsicCalibrations(existingCalibrationDirectory);
    }
    calibrator->InitialiseVideo();

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

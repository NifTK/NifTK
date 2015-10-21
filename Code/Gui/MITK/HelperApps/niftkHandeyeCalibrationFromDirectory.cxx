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

  bool sortByDistance = !dontSortByDistance;
  try
  {
    mitk::Point2D pixelScales;
    pixelScales[0] = pixelScaleFactors[0];
    pixelScales[1] = pixelScaleFactors[1];

    mitk::HandeyeCalibrateFromDirectory::Pointer calibrator = mitk::HandeyeCalibrateFromDirectory::New();
    calibrator->SetInputDirectory(trackingInputDirectory);
    calibrator->SetOutputDirectory(outputDirectory);
    calibrator->SetTrackerIndex(trackerIndex);
    calibrator->SetAbsTrackerTimingError(maxTimingError);
    calibrator->SetFramesToUse(framesToUse);
    calibrator->SetFramesToUseFactor(framesToUseFactor);
    calibrator->SetSortByDistance(sortByDistance);
    calibrator->SetFlipTracking(flipTracking);
    calibrator->SetFlipExtrinsic(flipExtrinsic);
    calibrator->SetSortByAngle(false);
    calibrator->SetPixelScaleFactor(pixelScales);
    calibrator->SetSwapVideoChannels(swapVideoChannels);
    calibrator->SetNumberCornersWidth(numberCornerWidth);
    calibrator->SetNumberCornersHeight(numberCornerHeight);
    calibrator->SetSquareSizeInMillimetres(squareSizeInMM);
    calibrator->SetRandomise(randomise);
    calibrator->InitialiseOutputDirectory();
    calibrator->InitialiseTracking();


    if ( existingIntrinsicsDirectory != "" )
    {
      MITK_INFO << "Attempting to use existing intrinsic calibration from " << existingIntrinsicsDirectory;
      calibrator->LoadExistingIntrinsicCalibrations(existingIntrinsicsDirectory);
    }

    if ( existingRightToLeftDirectory != "" )
    {
      MITK_INFO << "Attempting to use existing right-to-left calibration from " << existingRightToLeftDirectory;
      calibrator->LoadExistingRightToLeft(existingRightToLeftDirectory);
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

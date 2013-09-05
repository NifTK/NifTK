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
  std::vector<double> ReprojectionError;

  try
  {
    mitk::HandeyeCalibrateFromDirectory::Pointer Calibrator = mitk::HandeyeCalibrateFromDirectory::New();
    Calibrator->SetDirectory(trackingInputDirectory);
    Calibrator->SetTrackerIndex(trackerIndex);
    Calibrator->SetAbsTrackerTimingError(MaxTimingError);
    Calibrator->SetFramesToUse(FramesToUse);
    Calibrator->SetSortByDistance(SortByDistance);
    Calibrator->SetFlipTracking(FlipTracking);
    Calibrator->SetFlipExtrinsic(FlipExtrinsic);
    Calibrator->SetSortByAngle(false);
    Calibrator->InitialiseTracking();
    Calibrator->InitialiseVideo();

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

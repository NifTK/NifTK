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
#include <mitkOpenCVFileIOUtils.h>
#include <mitkTrackingAndTimeStampsContainer.h>
#include <mitkExceptionMacro.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if ( matrixDirectory.length() == 0
       || outputMatrixFile.length() == 0
       || pointDirectory.length() == 0
       )
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    std::cout << "niftkUltrasoundPinCalibration: matrices       = " << matrixDirectory << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: points         = " << pointDirectory << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: output         = " << outputMatrixFile << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: opt scaling    = " << optimiseScaling << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: mm/pix         = " << millimetresPerPixel[0] << ", " << millimetresPerPixel[1] << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: opt inv point  = " << optimiseInvariantPoint << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: inv point      = " << invariantPoint[0] << ", " << invariantPoint[1] << ", " << invariantPoint[2] << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: opt timing lag = " << optimiseTimingLag << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: timing lag     = " << timingLag << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: initial guess  = " << initialGuess << std::endl;

    mitk::Point2D mmPerPix;
    mmPerPix[0] = millimetresPerPixel[0];
    mmPerPix[1] = millimetresPerPixel[1];

    mitk::Point3D invPoint;
    invPoint[0] = invariantPoint[0];
    invPoint[1] = invariantPoint[1];
    invPoint[2] = invariantPoint[2];

    mitk::UltrasoundPinCalibration::Pointer calibration = mitk::UltrasoundPinCalibration::New();
    calibration->SetOptimiseImageScaleFactors(optimiseScaling);
    calibration->SetImageScaleFactors(mmPerPix);
    calibration->SetOptimiseInvariantPoint(optimiseInvariantPoint);
    calibration->SetInvariantPoint(invPoint);
    calibration->SetOptimiseTimingLag(optimiseTimingLag);
    calibration->SetTimingLag(timingLag);
    calibration->LoadRigidTransformation(initialGuess);

    mitk::TrackingAndTimeStampsContainer trackingData;
    trackingData.LoadFromDirectory(matrixDirectory);
    if (trackingData.GetSize() == 0)
    {
      mitkThrow() << "Failed to tracking data from " << matrixDirectory << std::endl;
    }
    calibration->SetTrackingData(&trackingData);

    std::vector< std::pair<unsigned long long, cv::Point3d> > pointData = mitk::LoadTimeStampedPoints(pointDirectory);
    if (pointData.size() == 0)
    {
      mitkThrow() << "Failed to load point data from " << pointDirectory << std::endl;
    }
    calibration->SetPointData(&pointData);

    double residualError = calibration->Calibrate();
    calibration->SaveRigidTransformation(outputMatrixFile);

    std::cout << "niftkUltrasoundPinCalibration: residual = " << residualError << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: scaling  = " << calibration->GetImageScaleFactors() << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: lag      = " << calibration->GetTimingLag() << " (ms) " << std::endl;

    returnStatus = EXIT_SUCCESS;
  }
  catch (mitk::Exception& e)
  {
    MITK_ERROR << "Caught mitk::Exception: " << e.GetDescription() << ", from:" << e.GetFile() << "::" << e.GetLine() << std::endl;
    returnStatus = EXIT_FAILURE + 1;
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception: " << e.what() << std::endl;
    returnStatus = EXIT_FAILURE + 2;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:" << std::endl;
    returnStatus = EXIT_FAILURE + 3;
  }

  return returnStatus;
}

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
#include <mitkExceptionMacro.h>
#include <mitkVector.h>

#include <mitkVideoHandEyeCalibration.h>
#include <mitkOpenCVFileIOUtils.h>
#include <mitkTrackingAndTimeStampsContainer.h>
#include <mitkFindAndTriangulateCrossHair.h>
#include <niftkVideoHandEyeByCrossCalibrationCLP.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if (    matrixDirectory.length() == 0
       || outputMatrixFile.length() == 0
       || (inputVideoData.length() == 0 && videoDirectory.length() == 0)
       )
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    typedef mitk::TimeStampsContainer::TimeStamp TimeStampType;

    std::cout << "niftkVideoHandEyeByCrossCalibration: matrices       = " << matrixDirectory << std::endl;
    std::cout << "niftkVideoHandEyeByCrossCalibration: video          = " << videoDirectory << std::endl;
    std::cout << "niftkVideoHandEyeByCrossCalibration: output         = " << outputMatrixFile << std::endl;
    std::cout << "niftkVideoHandEyeByCrossCalibration: opt inv point  = " << optimiseInvariantPoint << std::endl;
    std::cout << "niftkVideoHandEyeByCrossCalibration: inv point      = " << invariantPoint[0] << ", " << invariantPoint[1] << ", " << invariantPoint[2] << std::endl;
    std::cout << "niftkVideoHandEyeByCrossCalibration: opt timing lag = " << optimiseTimingLag << std::endl;
    std::cout << "niftkVideoHandEyeByCrossCalibration: timing lag     = " << timingLag << "ms" <<  std::endl;
    std::cout << "niftkVideoHandEyeByCrossCalibration: max timing Error     = " << maxTimingError << "ms" <<  std::endl;
    std::cout << "niftkVideoHandEyeByCrossCalibration: initial guess  = " << initialGuess << std::endl;

    mitk::Point3D invPoint;
    invPoint[0] = invariantPoint[0];
    invPoint[1] = invariantPoint[1];
    invPoint[2] = invariantPoint[2];

    mitk::VideoHandEyeCalibration::Pointer calibration = mitk::VideoHandEyeCalibration::New();
    calibration->SetOptimiseInvariantPoint(optimiseInvariantPoint);
    calibration->SetInvariantPoint(invPoint);
    calibration->SetOptimiseTimingLag(optimiseTimingLag);
    calibration->SetTimingLag(timingLag * 1e-3);
    calibration->SetAllowableTimingError(maxTimingError * 1e6);
    calibration->LoadRigidTransformation(initialGuess);
    calibration->SetVerbose(verbose);

    mitk::TrackingAndTimeStampsContainer trackingData;
    trackingData.LoadFromDirectory(matrixDirectory);
    if (trackingData.GetSize() == 0)
    {
      mitkThrow() << "Failed to tracking data from " << matrixDirectory << std::endl;
    }
    calibration->SetTrackingData(&trackingData);

    // Here we start creating point data, and we load into this data structure.
    std::vector< std::pair<TimeStampType, cv::Point3d> > pointData;

    // If we specify a video directory, we extract the points from the video.
    if (videoDirectory.length() != 0)
    {
      mitk::FindAndTriangulateCrossHair::Pointer triangulator = mitk::FindAndTriangulateCrossHair::New();
      triangulator->Initialise(videoDirectory, cameraCalibrationDirectory);
      triangulator->SetTrackerIndex(0);
      triangulator->SetFlipMatrices(false);
      triangulator->SetVisualise(true);
      triangulator->Triangulate();

      std::vector<mitk::ProjectedPointPair> projectedPoints = triangulator->GetScreenPoints();
      std::vector<mitk::WorldPoint> cameraPoints = triangulator->GetPointsInLeftLensCS();
      assert(projectedPoints.size() == cameraPoints.size());

      for (unsigned long int i = 0; i < projectedPoints.size(); i++)
      {
        if (!projectedPoints[i].LeftNaNOrInf() && !projectedPoints[i].RightNaNOrInf())
        {
          pointData.push_back(std::pair<TimeStampType, cv::Point3d>(projectedPoints[i].m_TimeStamp, cameraPoints[i].m_Point));
        }
      }

      if (pointData.size() == 0)
      {
        mitkThrow() << "Failed to load point data from directory " << videoDirectory << std::endl;
      }

      // Write to file, so we don't have to keep running the above extraction process.
      // (much as I like watching the live cross detection :-)
      if (outputVideoData.length() > 0)
      {
        mitk::SaveTimeStampedPoints(pointData, outputVideoData);
      }

    } else if (inputVideoData.length() > 0)
    {
      mitk::LoadTimeStampedPoints(pointData, inputVideoData);
    }

    calibration->SetPointData(&pointData);

    std::cout << "niftkVideoHandEyeByCrossCalibration: number of points = " << pointData.size() << std::endl;
    std::cout << "niftkVideoHandEyeByCrossCalibration: number of matrices = " << trackingData.GetSize() << std::endl;

    // Now we can do calibration.
    double residualError = calibration->Calibrate();
    calibration->SaveRigidTransformation(outputMatrixFile);

    std::cout << "niftkVideoHandEyeByCrossCalibration: residual  = " << residualError << std::endl;
    std::cout << "niftkVideoHandEyeByCrossCalibration: lag       = " << calibration->GetTimingLag() << " (seconds) " << std::endl;
    std::cout << "niftkVideoHandEyeByCrossCalibration: inv point = " << calibration->GetInvariantPoint() << std::endl;

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

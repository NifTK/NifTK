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

#include <mitkProjectPointsOnStereoVideo.h>
#include <mitkOpenCVMaths.h>
#include <mitkOpenCVPointTypes.h>
#include <mitkOpenCVFileIOUtils.h>
#include <mitkIOUtil.h>
#include <niftkCalculateReProjectionErrorsCLP.h>
#include <boost/lexical_cast.hpp>

#include <fstream>
int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if ( trackingInputDirectory.length() == 0 )
  {
    std::cout << trackingInputDirectory.length() << std::endl;
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  if ( calibrationInputDirectory.length() == 0 )
  {
    std::cout << calibrationInputDirectory.length() << std::endl;
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  if ( input3DDirectory.length() == 0 || goldStandardDirectory.length() == 0 )
  {
    std::cout << "no point input files defined " << std::endl;
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    mitk::ProjectPointsOnStereoVideo::Pointer projector = mitk::ProjectPointsOnStereoVideo::New();
    bool Visualise = false;
    float projectorScreenBuffer = 0.0;
    float classifierScreenBuffer = 100.0;
    bool FlipTracking = false;
    bool WriteTimingErrors = false;
    bool DrawAxes = false;
    bool outputVideo = false;
    bool showTrackingStatus = false;
    bool annotateWithGS = false;
    int referenceIndex = -1;
    float pointMatchingRatio = 3.0;

    projector->SetVisualise(Visualise);
    projector->SetAllowableTimingError(maxTimingError * 1e6);
    projector->SetProjectorScreenBuffer(projectorScreenBuffer);
    projector->SetClassifierScreenBuffer(classifierScreenBuffer);
    projector->SetVisualiseTrackingStatus(showTrackingStatus);
    if ( saveImages )
    {
      annotateWithGS = true;
    }
    projector->SetAnnotateWithGoldStandards(annotateWithGS);
    projector->SetWriteAnnotatedGoldStandards(saveImages);

    if ( outputVideo )
    {
      projector->SetSaveVideo(true);
    }
    projector->Initialise(trackingInputDirectory,calibrationInputDirectory);
    mitk::VideoTrackerMatching::Pointer matcher = mitk::VideoTrackerMatching::New();
    matcher->Initialise(trackingInputDirectory);
    if ( videoLag != 0 )
    {
      if ( videoLag < 0 )
      {
        matcher->SetVideoLagMilliseconds(videoLag,true);
      }
      else
      {
        matcher->SetVideoLagMilliseconds(videoLag,false);
      }
    }

    if ( ! projector->GetInitOK() )
    {
      MITK_ERROR << "Projector failed to initialise, halting.";
      return -1;
    }
    matcher->SetFlipMatrices(FlipTracking);
    matcher->SetWriteTimingErrors(WriteTimingErrors);
    projector->SetTrackerIndex(trackerIndex);
    projector->SetReferenceIndex(referenceIndex);
    projector->SetMatcherCameraToTracker(matcher);
    projector->SetDrawAxes(DrawAxes);

    std::vector < mitk::ProjectedPointPair > screenPoints;
    std::vector < unsigned int  > screenPointFrameNumbers;
    std::vector < mitk::WorldPoint > worldPoints;
    std::vector < mitk::WorldPoint > classifierWorldPoints;
    std::vector < mitk::WorldPoint > worldPointsWithScalars;
    if ( input3DDirectory.length() != 0 )
    {
      projector->SetModelPoints ( mitk::LoadPickedPointListFromDirectoryOfMPSFiles ( input3DDirectory ));
    }
    if ( modelToWorld.length() != 0 )
    {
      cv::Mat* modelToWorldMat = new cv::Mat(4,4,CV_64FC1);
      if ( mitk::ReadTrackerMatrix(modelToWorld, *modelToWorldMat) )
      {
        projector->SetModelToWorldTransform ( modelToWorldMat );
      }
      else
      {
        MITK_ERROR << "Failed to read mode to world file " << modelToWorld << ", halting";
        return EXIT_FAILURE;
      }
    }

    if ( goldStandardDirectory.length() != 0 )
    {
      std::vector < mitk::PickedObject > pickedObjects;
      mitk::LoadPickedObjectsFromDirectory ( pickedObjects, goldStandardDirectory );
      projector->SetGoldStandardObjects (pickedObjects);
    }

    projector->Project(matcher);

    if ( outputFile.length() != 0 )
    {
      projector->SetAllowablePointMatchingRatio(pointMatchingRatio);
      bool useNewOutputFormat = true;
      projector->CalculateProjectionErrors(outputFile, useNewOutputFormat);
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

  return returnStatus;
}

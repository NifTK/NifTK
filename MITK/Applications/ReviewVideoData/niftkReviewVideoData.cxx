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
#include <niftkReviewVideoDataCLP.h>

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


  try
  {
    mitk::ProjectPointsOnStereoVideo::Pointer projector = mitk::ProjectPointsOnStereoVideo::New();
    projector->SetVisualise(! noVisualise);
    projector->SetAllowableTimingError(maxTimingError * 1e6);
    projector->SetDontProject(true);
    projector->SetVisualiseTrackingStatus(true);
    projector->SetFlipVideo(flipVideo);
    
    if ( outputVideo ) 
    {
      projector->SetSaveVideo(true);
    }
    if ( writeTrackingMatricesPerFrame )
    {
      projector->SetWriteTrackingMatrixFilesPerFrame(true);
    }
    if ( dontCorrectAspectRatio )
    {
      projector->SetCorrectVideoAspectRatioByHalvingWidth(false);
    }
    projector->Initialise(trackingInputDirectory);
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
    matcher->SetFlipMatrices(false);
    matcher->SetWriteTimingErrors(WriteTimingErrors);
   

    projector->Project(matcher);

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

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

#include <mitkVideoToSurface.h>
#include <mitkOpenCVFileIOUtils.h>
#include <mitkIOUtil.h>
#include <niftkVideoToSurfacesCLP.h>
#include <boost/lexical_cast.hpp>

#include <fstream>
int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if ( videoAndTrackingDirectory.length() == 0 )
  {
    std::cout << "No tracking input directory " << videoAndTrackingDirectory.length() << std::endl;
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  if ( calibrationDirectory.length() == 0 )
  {
    std::cout << "No calibration directory " << calibrationDirectory.length() << std::endl;
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    mitk::VideoToSurface::Pointer projector = mitk::VideoToSurface::New();
    projector->SetEndFrame(endFrame);
   // projector->SetAllowableTimingError(maxTimingError * 1e6);
    
    projector->SetSaveVideo(true);
    projector->Initialise(videoAndTrackingDirectory,calibrationDirectory);
    mitk::VideoTrackerMatching::Pointer matcher = mitk::VideoTrackerMatching::New();
    matcher->Initialise(videoAndTrackingDirectory);

    if ( ! projector->GetInitOK() ) 
    {
      MITK_ERROR << "Projector failed to initialise, halting.";
      return -1;
    }
    projector->SetTrackerIndex(trackerIndex);
    projector->SetMatcherCameraToTracker(matcher);

    projector->Reconstruct(matcher);
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

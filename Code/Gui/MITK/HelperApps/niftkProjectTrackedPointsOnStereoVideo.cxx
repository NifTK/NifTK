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
#include <niftkProjectTrackedPointsOnStereoVideoCLP.h>

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

  if ( input2D.length() == 0 && input3D.length() )
  {
    std::cout << "no point input files defined " << std::endl;
    commandLine.getOutput()->usage(commandLine);
  }

  try
  {
    mitk::ProjectPointsOnStereoVideo::Pointer projector = mitk::ProjectPointsOnStereoVideo::New();
    projector->SetVisualise(Visualise);
    projector->Initialise(trackingInputDirectory,calibrationInputDirectory);
    if ( ! projector->GetInitOK() ) 
    {
      MITK_ERROR << "Projector failed to initialise, halting.";
      return -1;
    }
    projector->SetFlipMatrices(FlipTracking);
    projector->SetTrackerIndex(trackerIndex);
    projector->SetDrawAxes(DrawAxes);
    
    std::vector < std::pair < cv::Point2f, cv::Point2f > > screenPoints;
    unsigned int setPointsFrameNumber;
    std::vector < cv::Point3f > worldPoints;
    if ( input2D.length() != 0 ) 
    {
      std::ifstream fin(input2D.c_str());
      fin >> setPointsFrameNumber;
      float x1;
      float y1;
      float x2;
      float y2;
      while ( fin >> x1 >> y1 >> x2 >> y2 )
      {
        screenPoints.push_back(std::pair<cv::Point2f,cv::Point2f> (cv::Point2f(x1,y1), cv::Point2f(x2,y2)));
      }
      fin.close();
      projector->SetWorldPointsByTriangulation(screenPoints,setPointsFrameNumber);
    }
  if ( input3D.length() != 0 ) 
    {
      std::ifstream fin(input3D.c_str());
      float x;
      float y;
      float z;
      while ( fin >> x >> y >> z  )
      {
        worldPoints.push_back(cv::Point3f(x,y,z));
      }
      projector->SetWorldPoints(worldPoints);
      fin.close();
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

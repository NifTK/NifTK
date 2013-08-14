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
#include <mitkFindAndTriangulateCrossHair.h>
#include <mitkOpenCVMaths.h>
#include <niftkFindAndTriangulateCrossHairCLP.h>

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


  try
  {
    mitk::FindAndTriangulateCrossHair::Pointer finder = mitk::FindAndTriangulateCrossHair::New();
    finder->SetVisualise(Visualise);
    finder->Initialise(trackingInputDirectory,calibrationInputDirectory);
    if ( ! finder->GetInitOK() ) 
    {
      MITK_ERROR << "Finder failed to initialise, halting.";
      return -1;
    }
    finder->SetFlipMatrices(FlipTracking);
    finder->SetTrackerIndex(trackerIndex);

    finder->Triangulate();
   
    std::vector < cv::Point3f > worldPoints = finder->GetWorldPoints();
    cv::Point3f worldCentroid;
    cv::Point3f* worldStdDev = new cv::Point3f;
    worldCentroid = mitk::GetCentroid (worldPoints, true, worldStdDev);
    MITK_INFO << "World centre = " << worldCentroid;
    MITK_INFO << "World std dev = " << *worldStdDev;
    if ( outputWorld.length() != 0 ) 
    {
      std::ofstream fout (outputWorld.c_str());
      fout << "#Frame Number " ;
      fout << "Pworld" << "x,y,z";
      fout << std::endl;
      for ( unsigned int i  = 0 ; i < worldPoints.size() ; i ++ )
      {
        fout << i << " ";
        fout << worldPoints[i].x << " " <<  worldPoints[i].y <<
             " " << worldPoints[i].z;
        fout << std::endl;
      }
      fout.close();
    }
    if ( outputLens.length() !=0 )
    {
      std::ofstream fout (outputLens.c_str());
      std::vector < cv::Point3f >  leftLensPoints = finder->GetPointsInLeftLensCS();
      fout << "#Frame Number " ;
      fout << "PleftLens" <<"[x,y,z]";
      fout << std::endl;
      for ( unsigned int i  = 0 ; i < leftLensPoints.size() ; i ++ )
      {
        fout << i << " ";
        fout << leftLensPoints[i].x << " " <<  leftLensPoints[i].y <<
             " " << leftLensPoints[i].z << " " ;
        fout << std::endl;
      }
      fout.close();
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

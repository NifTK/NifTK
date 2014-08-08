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
    finder->SetFramesToProcess (framesToUse);
    finder->Initialise(trackingInputDirectory,calibrationInputDirectory);
    if ( videoLag != 0 )
    {
      if ( videoLag < 0 )
      {
        finder->SetVideoLagMilliseconds(videoLag,true);
      } 
      else
      {
        finder->SetVideoLagMilliseconds(videoLag,false);
      } 
    } 
    if ( ! finder->GetInitOK() ) 
    {
      MITK_ERROR << "Finder failed to initialise, halting.";
      return -1;
    }
    finder->SetFlipMatrices(FlipTracking);
    finder->SetTrackerIndex(trackerIndex);

    finder->Triangulate();
   
    std::vector < mitk::WorldPoint > mitkWorldPoints = finder->GetWorldPoints();
    std::vector < cv::Point3d > worldPoints;

    for ( unsigned int i = 0 ; i < mitkWorldPoints.size() ; i ++ ) 
    {
      worldPoints.push_back (mitkWorldPoints[i].m_Point);
    }
    cv::Point3d worldCentroid;
    cv::Point3d* worldStdDev = new cv::Point3d;
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
      std::vector < mitk::WorldPoint >  leftLensPoints = finder->GetPointsInLeftLensCS();
      std::vector < mitk::ProjectedPointPair > screenPoints = finder->GetScreenPoints();
      fout << "#Frame Number " ;
      fout << "PleftLens" << "[x,y,z]" << "PLeftScreen [x,y] , PRightScreen [x,y]";
      fout << std::endl;
      for ( unsigned int i  = 0 ; i < leftLensPoints.size() ; i ++ )
      {
        fout << i << " ";
        fout << leftLensPoints[i].m_Point.x << " " <<  leftLensPoints[i].m_Point.y <<
             " " << leftLensPoints[i].m_Point.z << " " <<
             screenPoints[i].m_Left.x << " " << screenPoints[i].m_Left.y << " " <<
             screenPoints[i].m_Right.x << " " << screenPoints[i].m_Right.y ;
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

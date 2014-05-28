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

  if ( input2D.length() == 0 && input3D.length() == 0 && input3DWithScalars.length() == 0  )
  {
    std::cout << "no point input files defined " << std::endl;
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    mitk::ProjectPointsOnStereoVideo::Pointer projector = mitk::ProjectPointsOnStereoVideo::New();
    projector->SetVisualise(Visualise);
    projector->SetAllowableTimingError(maxTimingError * 1e6);
    projector->SetProjectorScreenBuffer(projectorScreenBuffer);
    projector->SetClassifierScreenBuffer(classifierScreenBuffer);
    
    if ( outputVideo.length() != 0 ) 
    {
      projector->SetSaveVideo(true, outputVideo);
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
    projector->SetTrackerIndex(trackerIndex);
    projector->SetReferenceIndex(referenceIndex);
    projector->SetMatcherCameraToTracker(matcher);
    projector->SetDrawAxes(DrawAxes);
    
    std::vector < std::pair < cv::Point2d, cv::Point2d > > screenPoints;
    std::vector < unsigned int  > screenPointFrameNumbers;
    std::vector < cv::Point3d > worldPoints;
    std::vector < cv::Point3d > classifierWorldPoints;
    std::vector < std::pair < cv::Point3d , cv::Scalar > > worldPointsWithScalars;
    if ( input2D.length() != 0 ) 
    {
      std::ifstream fin(input2D.c_str());
      unsigned int frameNumber;
      double x1;
      double y1;
      double x2;
      double y2;
      while ( fin >> frameNumber >> x1 >> y1 >> x2 >> y2 )
      {
        screenPoints.push_back(std::pair<cv::Point2d,cv::Point2d> (cv::Point2d(x1,y1), cv::Point2d(x2,y2)));
        screenPointFrameNumbers.push_back(frameNumber);
      }
      fin.close();
      projector->SetWorldPointsByTriangulation(screenPoints,screenPointFrameNumbers,matcher);
    }
    if ( input3D.length() != 0 ) 
    {
      std::ifstream fin(input3D.c_str());
      double x;
      double y;
      double z;
      while ( fin >> x >> y >> z  )
      {
        worldPoints.push_back(cv::Point3d(x,y,z));
      }
      projector->SetWorldPoints(worldPoints);
      fin.close();
    }
   
    if ( classifier3D.length() != 0 ) 
    {
      std::ifstream fin(classifier3D.c_str());
      double x;
      double y;
      double z;
      while ( fin >> x >> y >> z  )
      {
        classifierWorldPoints.push_back(cv::Point3d(x,y,z));
      }
      projector->SetClassifierWorldPoints(classifierWorldPoints);
      fin.close();
    }
   

    if ( input3DWithScalars.length() != 0 ) 
    {
      std::ifstream fin(input3DWithScalars.c_str());
      double x;
      double y;
      double z;
      int b; 
      int g;
      int r;

      while ( fin >> x >> y >> z >> b >> g >> r )
      {
        worldPointsWithScalars.push_back(std::pair < cv::Point3d,cv::Scalar > 
            (cv::Point3d(x,y,z), cv::Scalar(r,g,b) ));
      }
      projector->SetWorldPoints(worldPointsWithScalars);
      fin.close();
    }


    projector->Project(matcher);
   
    if ( output2D.length() != 0 ) 
    {
      std::ofstream fout (output2D.c_str());
      std::vector < std::pair < long long , std::vector < std::pair < cv::Point2d , cv::Point2d > > > > projectedPoints = 
        projector->GetProjectedPoints();
      fout << "#Frame Number " ;
      for ( unsigned int i = 0 ; i < projectedPoints[0].second.size() ; i ++ ) 
      {
        fout << "P" << i << "[lx,ly,rx,ry]" << " ";
      }
      fout << std::endl;
      for ( unsigned int i  = 0 ; i < projectedPoints.size() ; i ++ )
      {
        fout << i << " ";
        for ( unsigned int j = 0 ; j < projectedPoints[i].second.size() ; j ++ )
        {
          fout << projectedPoints[i].second[j].first.x << " " <<  projectedPoints[i].second[j].first.y <<
             " " << projectedPoints[i].second[j].second.x << " " << projectedPoints[i].second[j].second.y << " ";
        }
        fout << std::endl;
      }
      fout.close();
    }
    if ( output3D.length() !=0 )
    {
      std::ofstream fout (output3D.c_str());
      std::vector < std::vector < cv::Point3d > > leftLensPoints = 
        projector->GetPointsInLeftLensCS();
      fout << "#Frame Number " ;
      for ( unsigned int i = 0 ; i < leftLensPoints[0].size() ; i ++ ) 
      {
        fout << "P" << i << "[x,y,z]" << " ";
      }
      fout << std::endl;
      for ( unsigned int i  = 0 ; i < leftLensPoints.size() ; i ++ )
      {
        fout << i << " ";
        for ( unsigned int j = 0 ; j < leftLensPoints[i].size() ; j ++ )
        {
          fout << leftLensPoints[i][j].x << " " <<  leftLensPoints[i][j].y <<
             " " << leftLensPoints[i][j].z << " " ;
        }
        fout << std::endl;
      }
      fout.close();
    }
    if ( outputMatrices.length() !=0 )
    {
      std::ofstream fout (outputMatrices.c_str());
      std::vector < cv::Mat > leftCameraMatrices = 
        projector->GetWorldToLeftCameraMatrices();
      
      for ( unsigned int i  = 0 ; i < leftCameraMatrices.size() ; i ++ )
      {
        fout << "#Frame " << i << std::endl;
        fout << leftCameraMatrices[i] ;
        fout << std::endl;
      }
      fout.close();
    }
    if ( leftGoldStandard.length() != 0 ) 
    {
      std::ifstream fin(leftGoldStandard.c_str());
      std::vector < mitk::GoldStandardPoint > leftGS;
      unsigned int frameNumber;
      cv::Point2d point;
      while ( fin >> frameNumber >> point.x >> point.y )
      {
        leftGS.push_back( mitk::GoldStandardPoint (frameNumber,-1,point));
      }
      fin.close();
      projector->SetLeftGoldStandardPoints(leftGS);
    }
    if ( rightGoldStandard.length() != 0 ) 
    {
      std::ifstream fin(rightGoldStandard.c_str());
      std::vector < mitk::GoldStandardPoint > rightGS;
      unsigned int frameNumber;
      cv::Point2d point;
      while ( fin >> frameNumber >> point.x >> point.y )
      {
        rightGS.push_back(mitk::GoldStandardPoint (frameNumber,-1,point));
      }
      fin.close();
      projector->SetRightGoldStandardPoints(rightGS);
    }

    if ( outputErrors.length() != 0 ) 
    {
      projector->SetAllowablePointMatchingRatio(pointMatchingRatio);
      projector->CalculateProjectionErrors(outputErrors);
      projector->CalculateTriangulationErrors(outputErrors, matcher);
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

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
#include <niftkProjectTrackedPointsOnStereoVideoCLP.h>
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

  if ( ( input2D.length() == 0 ) && 
       ( input3D.length() == 0 ) && 
       ( input3DWithScalars.length() == 0 ) && 
       ( leftGoldStandard.length() == 0 || rightGoldStandard.length() == 0 ) &&
       ( goldStandardObjects.length() == 0 ) )
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
    projector->SetVisualiseTrackingStatus(showTrackingStatus);
    if ( saveAnnotateWithGS )
    {
      annotateWithGS = true;
    }
    projector->SetAnnotateWithGoldStandards(annotateWithGS);
    projector->SetWriteAnnotatedGoldStandards(saveAnnotateWithGS);
    
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
        screenPoints.push_back( mitk::ProjectedPointPair (cv::Point2d(x1,y1), cv::Point2d(x2,y2)));
        screenPointFrameNumbers.push_back(frameNumber);
      }
      fin.close();
      projector->AppendWorldPointsByTriangulation(screenPoints,screenPointFrameNumbers,matcher);
    }
    if ( input3D.length() != 0 ) 
    {
      //try reading it as a mitk point set first
      mitk::PointSet::Pointer pointSet = mitk::IOUtil::LoadPointSet ( input3D ); 
      if ( pointSet->GetSize() == 0 ) 
      {
        //try reading a stream of points instead
        std::ifstream fin(input3D.c_str());
        double x;
        double y;
        double z;
        while ( fin >> x >> y >> z  )
        {
          worldPoints.push_back(mitk::WorldPoint(cv::Point3d(x,y,z)));
        }
        fin.close();
      }
      else
      {
        std::vector < cv::Point3d > worldPointsVector = mitk::PointSetToVector ( pointSet );
        for ( unsigned int i = 0 ; i < worldPointsVector.size() ; i ++ ) 
        {
          worldPoints.push_back ( mitk::WorldPoint(worldPointsVector[i] ) );
        }
      }
      projector->AppendWorldPoints(worldPoints);
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
      projector->AppendClassifierWorldPoints(classifierWorldPoints);
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
      
      std::string in[6];
      while ( fin >> in[0] >> in[1] >> in[2] >> in[3] >> in[4] >> in[5] )
      {
        x=atof(in[0].c_str());
        y=atof(in[1].c_str());
        z=atof(in[2].c_str());
        b=atoi(in[3].c_str());
        g=atoi(in[4].c_str());
        r=atoi(in[5].c_str());

        worldPointsWithScalars.push_back(mitk::WorldPoint 
            (cv::Point3d(x,y,z), cv::Scalar(r,g,b) ));
      }
      projector->AppendWorldPoints(worldPointsWithScalars);
      fin.close();
    }
   
    if ( goldStandardObjects.length() != 0 ) 
    {
      std::ifstream fin ( goldStandardObjects.c_str() );
      if ( fin ) 
      {
        std::vector < mitk::PickedObject > pickedObjects;
        mitk::LoadPickedObjects ( pickedObjects, fin );
        projector->SetGoldStandardObjects (pickedObjects);
        fin.close();
      }
      else
      {
        MITK_ERROR << "Failed to open " << goldStandardObjects << " for input";
      }
    }
    if ( leftGoldStandard.length() != 0 ) 
    {
      std::ifstream fin(leftGoldStandard.c_str());
      std::vector < mitk::GoldStandardPoint > leftGS;
      while ( fin  )
      {
        mitk::GoldStandardPoint point(fin);
        if ( fin )
        {
          leftGS.push_back( point );
        }
      }
      fin.close();
      projector->SetLeftGoldStandardPoints(leftGS, matcher);
    }
    if ( rightGoldStandard.length() != 0 ) 
    {
      std::ifstream fin(rightGoldStandard.c_str());
      std::vector < mitk::GoldStandardPoint > rightGS;
      while ( fin )
      {
        mitk::GoldStandardPoint point(fin);
        if ( fin ) 
        {
          rightGS.push_back(point);
        }
      }
      fin.close();
      projector->SetRightGoldStandardPoints(rightGS, matcher);
    }

    projector->Project(matcher);
   
    if ( output2D.length() != 0 ) 
    {
      std::ofstream fout (output2D.c_str());
      std::vector < mitk::ProjectedPointPairsWithTimingError > projectedPoints = 
        projector->GetProjectedPoints();
      fout << "#Frame Number " ;
      for ( unsigned int i = 0 ; i < projectedPoints[0].m_Points.size() ; i ++ ) 
      {
        fout << "P" << i << "[lx,ly,rx,ry]" << " ";
      }
      fout << std::endl;
      for ( unsigned int i  = 0 ; i < projectedPoints.size() ; i ++ )
      {
        fout << i << " ";
        for ( unsigned int j = 0 ; j < projectedPoints[i].m_Points.size() ; j ++ )
        {
          fout << projectedPoints[i].m_Points[j].m_Left.x << " " <<  projectedPoints[i].m_Points[j].m_Left.y <<
             " " << projectedPoints[i].m_Points[j].m_Right.x << " " << projectedPoints[i].m_Points[j].m_Right.y << " ";
        }
        fout << std::endl;
      }
      fout.close();
    }
    if ( output3D )
    {
      std::vector <mitk::PickedPointList::Pointer> leftLensPoints = 
        projector->GetPointsInLeftLensCS();
      for ( unsigned int i  = 0 ; i < leftLensPoints.size() ; i ++ )
      {
        std::ofstream frameOut ( (boost::lexical_cast<std::string>(leftLensPoints[i]->GetFrameNumber()) + ".leftLensPoints").c_str() );
        if ( frameOut )
        {
          leftLensPoints[i]->PutOut(frameOut);
        }
        else
        {
          MITK_ERROR << "Failed to open out put file " << boost::lexical_cast<std::string>(leftLensPoints[i]->GetFrameNumber()) + ".leftLensPoints"; 
        }
        frameOut.close();
      }
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
    if ( outputTriangulatedPoints.length() != 0 )
    {
      projector->SetTriangulatedPointsOutName(outputTriangulatedPoints);
    }
    if ( outputErrors.length() != 0 ) 
    {
      projector->SetAllowablePointMatchingRatio(pointMatchingRatio);
      projector->CalculateProjectionErrors(outputErrors);
      projector->CalculateTriangulationErrors(outputErrors);
    }
    else
    {
      if ( outputTriangulatedPoints.length() != 0 )
      {
        projector->SetAllowablePointMatchingRatio(pointMatchingRatio);
        projector->TriangulateGoldStandardPoints(outputTriangulatedPoints, matcher);
      }
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

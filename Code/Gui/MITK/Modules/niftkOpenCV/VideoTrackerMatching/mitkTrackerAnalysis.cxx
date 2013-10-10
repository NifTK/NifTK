/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include "mitkTrackerAnalysis.h"
#include "mitkProjectPointsOnStereoVideo.h"
#include <mitkCameraCalibrationFacade.h>
#include <mitkUltrasoundPinCalibration.h>
#include <mitkOpenCVMaths.h>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include <sstream>
#include <fstream>
#include <cstdlib>

namespace mitk 
{
//---------------------------------------------------------------------------
TrackerAnalysis::TrackerAnalysis () 
{}

//---------------------------------------------------------------------------
TrackerAnalysis::~TrackerAnalysis () 
{}

//---------------------------------------------------------------------------
void TrackerAnalysis::TemporalCalibration(std::string calibrationfilename ,
    int windowLow, int windowHigh, bool visualise, std::string fileout)
{
  if ( !m_Ready )
  {
    MITK_ERROR << "Initialise video tracker matcher before attempting temporal calibration";
    return;
  }

  std::vector < std::ofstream* > fout;
  if ( fileout.length() != 0 ) 
  {
    for ( unsigned int i = 0 ; i <  m_TrackingMatrixTimeStamps.size() ; i ++ ) 
    {
      std::string thisfileout = fileout + boost::lexical_cast<std::string>(i) + ".txt";
      std::ofstream* thisfout = new std::ofstream();
      thisfout->open ( thisfileout.c_str() );
      if ( !thisfout )
      {
        MITK_WARN << "Failed to open output file for temporal calibration " << thisfileout;
      }
      fout.push_back(thisfout);
    }
  }

  std::vector < std::vector <cv::Point3d> >  pointsInLensCS;
  pointsInLensCS.clear();
  std::vector < std::vector < std::pair < cv::Point2d, cv::Point2d > > > onScreenPoints;
  onScreenPoints.clear();
  pointsInLensCS = ReadPointsInLensCSFile(calibrationfilename, 1 , &onScreenPoints);

  if ( pointsInLensCS.size() * 2 != m_FrameNumbers.size() )
  {
    MITK_ERROR << "Temporal calibration file has wrong number of frames, " << pointsInLensCS.size() * 2 << " != " << m_FrameNumbers.size() ;
    return;
  }

  mitk::ProjectPointsOnStereoVideo::Pointer projector = mitk::ProjectPointsOnStereoVideo::New();
  projector->Initialise(m_Directory,m_CalibrationDirectory);

  std::vector < std::vector < cv::Point3d > > reconstructedPointSD (m_TrackingMatrixTimeStamps.size());
  std::vector < std::vector < std::pair <double, double > > > projectedErrorRMS (m_TrackingMatrixTimeStamps.size());
  for ( int videoLag = windowLow; videoLag <= windowHigh ; videoLag ++ )
  {
    if ( videoLag < 0 ) 
    {
      SetVideoLagMilliseconds ( (unsigned long long) (videoLag * -1) , true, -1 );
    }
    else 
    {
      SetVideoLagMilliseconds ( (unsigned long long) (videoLag ) , false, -1  );
    }
   
    for ( unsigned int trackerIndex = 0 ; trackerIndex < m_TrackingMatrixTimeStamps.size() ; trackerIndex++ )
    {
      std::vector <cv::Point3d> worldPoints;
      worldPoints.clear();
      for ( unsigned int frame = 0 ; frame < pointsInLensCS.size() ; frame++ )
      {
        int framenumber = frame * 2;
        worldPoints.push_back (GetCameraTrackingMatrix(framenumber, NULL , trackerIndex ) *
            pointsInLensCS[frame][0]);
      }
      
      cv::Point3d pointSpread;
      cv::Point3d worldCentre = mitk::GetCentroid (worldPoints, true,  &pointSpread);

      reconstructedPointSD[trackerIndex].push_back(pointSpread);
      std::vector <cv::Point3d > worldPoint(1);
      worldPoint[0] = worldCentre;
      projector->SetWorldPoints(worldPoint);
      projector->SetTrackerIndex(trackerIndex);
      projector->Project(this);
      std::vector < std::vector < std::pair < cv::Point2d, cv::Point2d > > > projectedPoints = 
        projector->GetProjectedPoints();
      
      std::pair <double, double> projectedRMS = mitk::RMSError ( projectedPoints,  onScreenPoints );
      projectedErrorRMS[trackerIndex].push_back(projectedRMS);
      
    }
  }

  if ( fileout.length() != 0 ) 
  {
    for ( unsigned int trackerIndex = 0 ; trackerIndex < m_TrackingMatrixTimeStamps.size() ; trackerIndex++ )
    {
      *fout[trackerIndex] << "#lag SDx SDy SDz RMSLeft RMSRight" << std::endl;
      for ( int videoLag = windowLow; videoLag <= windowHigh ; videoLag ++ )
      {
        *fout[trackerIndex] << videoLag << " " << 
        reconstructedPointSD[trackerIndex][videoLag - windowLow].x << " " <<
        reconstructedPointSD[trackerIndex][videoLag - windowLow].y << " " <<
        reconstructedPointSD[trackerIndex][videoLag - windowLow].z << " " <<
        projectedErrorRMS[trackerIndex][videoLag - windowLow].first << " " <<
        projectedErrorRMS[trackerIndex][videoLag - windowLow].second << std::endl;
      }
      fout[trackerIndex]->close();
    }

  }

  for ( unsigned int trackerIndex = 0 ; trackerIndex < m_TrackingMatrixTimeStamps.size() ; trackerIndex++ )
  {
    std::pair < unsigned int , unsigned int > minIndexes;
    std::pair < double , double > minValues = mitk::FindMinimumValues ( projectedErrorRMS[trackerIndex], &minIndexes );
    MITK_INFO << "Tracker Index " << trackerIndex << " min left RMS " << minValues.first << " at " << (int)minIndexes.first + windowLow << " ms ";  ;
    MITK_INFO << "Tracker Index " << trackerIndex << " min right RMS " << minValues.second << " at " << (int)minIndexes.second + windowLow << " ms ";  ;
  }

}
//---------------------------------------------------------------------------
void TrackerAnalysis::OptimiseHandeyeCalibration(std::string calibrationfilename ,
    bool visualise, std::string fileout)
{
  MITK_ERROR << "TrackerAnalysis::OptimiseHandeyeCalibration is currently broken, do not use";
  return;
  if ( !m_Ready )
  {
    MITK_ERROR << "Initialise video tracker matcher before attempting temporal calibration";
    return;
  }

  std::ofstream fout;
  if ( fileout.length() != 0 ) 
  {
    fout.open(fileout.c_str());
    if ( !fout )
    {
      MITK_WARN << "Failed to open output file for temporal calibration " << fileout;
    }
  }

  std::vector < std::vector <cv::Point3d> > pointsInLensCS;
  pointsInLensCS.clear();
  std::vector <std::vector <std::pair <cv::Point2d, cv::Point2d > > >* onScreenPoints = new std::vector < std::vector <std::pair<cv::Point2d, cv::Point2d> > >;
  onScreenPoints->clear();
  pointsInLensCS = ReadPointsInLensCSFile(calibrationfilename, 1, onScreenPoints);

  if ( pointsInLensCS.size() * 2 != m_FrameNumbers.size() )
  {
    MITK_ERROR << "Temporal calibration file has wrong number of frames, " << pointsInLensCS.size() * 2 << " != " << m_FrameNumbers.size() ;
    return;
  }

  for ( unsigned int trackerIndex = 0 ; trackerIndex < m_TrackingMatrixTimeStamps.size() ; trackerIndex++ )
  {
    std::vector<cv::Mat> cameraMatrices;
    cameraMatrices.clear();
    for ( unsigned int frame = 0 ; frame < pointsInLensCS.size() ; frame++ )
    {
      int framenumber = frame * 2;
      if ( ! ( boost::math::isnan(pointsInLensCS[frame][0].x) || boost::math::isnan(pointsInLensCS[frame][0].y) || boost::math::isnan(pointsInLensCS[frame][0].z) ) )
      {
        cameraMatrices.push_back (GetCameraTrackingMatrix(framenumber, NULL , trackerIndex ));
      }
      else
      {
       pointsInLensCS.erase(pointsInLensCS.begin() + frame);
       frame -- ;
      }
    }
    //bool optimiseScaling = false;
    //bool optimiseInvariantPoint = true;
    std::vector<double> rigidBodyTransformation;
    cv::Point3d invariantPoint;
    //initial values for point and transform. Could use known handeye and reconstructed point, 
    //but for the minute let's use ID and 0 0 0 
    rigidBodyTransformation.clear();
    for ( int i = 0 ; i < 6 ; i ++ ) 
    {
      rigidBodyTransformation.push_back(0.0);
    }
    invariantPoint.x=0.0;
    invariantPoint.y=0.0;
    invariantPoint.z=0.0;
    cv::Point2d millimetresPerPixel;
    //mm per pixel has no meaning in this application as the point is already defined in mm
    millimetresPerPixel.x = 1.0;
    millimetresPerPixel.y = 1.0;

    cv::Matx44d outputMatrix;
    double residualError;
    
    mitk::UltrasoundPinCalibration::Pointer invPointCal = mitk::UltrasoundPinCalibration::New();
    //invPointCal->Calibrate(cameraMatrices, pointsInLensCS,
    //    optimiseScaling, optimiseInvariantPoint, rigidBodyTransformation,
    //    invariantPoint, millimetresPerPixel,
    //    outputMatrix, residualError);
    MITK_INFO << "Tracker Index " << trackerIndex << ": After optimisation, handeye = " ;
    for ( int i = 0 ; i < 4 ; i ++ )
    {
      MITK_INFO << outputMatrix(i,0) << "," << outputMatrix(i,1) <<  " , " << outputMatrix(i,2) << " , " <<  outputMatrix (i,3);
    }
    MITK_INFO << "Invariant point = " << invariantPoint << " [ " << residualError << " SD ].";
    MITK_INFO << "Millimetres per pixel = " << millimetresPerPixel;
  }

}
//---------------------------------------------------------------------------
void TrackerAnalysis::HandeyeSensitivityTest(std::string calibrationfilename ,
    double windowLow, double windowHigh , double stepSize, std::string fileout)
{
  if ( !m_Ready )
  {
    MITK_ERROR << "Initialise video tracker matcher before attempting handeye sensitivity";
    return;
  }

  std::vector < std::ofstream* > fout;
  if ( fileout.length() != 0 ) 
  {
    for ( unsigned int i = 0 ; i <  m_TrackingMatrixTimeStamps.size() ; i ++ ) 
    {
      std::string thisfileout = fileout + boost::lexical_cast<std::string>(i) + ".txt";
      std::ofstream* thisfout = new std::ofstream();
      thisfout->open ( thisfileout.c_str() );
      if ( !thisfout )
      {
        MITK_WARN << "Failed to open output file for handeye sensitivity " << thisfileout;
      }
      fout.push_back(thisfout);
    }
  }

  std::vector < std::vector <cv::Point3d> >  pointsInLensCS;
  pointsInLensCS.clear();
  std::vector < std::vector < std::pair < cv::Point2d, cv::Point2d > > > onScreenPoints;
  onScreenPoints.clear();
  pointsInLensCS = ReadPointsInLensCSFile(calibrationfilename, 1 , &onScreenPoints);

  if ( pointsInLensCS.size() * 2 != m_FrameNumbers.size() )
  {
    MITK_ERROR << "Handeye sensitivity file has wrong number of frames, " << pointsInLensCS.size() * 2 << " != " << m_FrameNumbers.size() ;
    return;
  }

  mitk::ProjectPointsOnStereoVideo::Pointer projector = mitk::ProjectPointsOnStereoVideo::New();
  projector->Initialise(m_Directory,m_CalibrationDirectory);

  std::vector < std::vector < cv::Point3d > > reconstructedPointSD (m_TrackingMatrixTimeStamps.size());
  std::vector < std::vector < std::pair <double, double > > > projectedErrorRMS (m_TrackingMatrixTimeStamps.size());
  std::vector < std::vector <double> > stateVector;
  for ( double tx = windowLow; tx <= windowHigh ; tx += stepSize )
  {
    MITK_INFO << "tx=" << tx;
    for ( double ty = windowLow; ty <= windowHigh ; ty += stepSize )
    {
      for ( double tz = windowLow; tz <= windowHigh ; tz += stepSize )
      {
        for ( double rx = windowLow; rx <= windowHigh ; rx += stepSize )
        {
          MITK_INFO << "rx="<< rx;
          for ( double ry = windowLow; ry <= windowHigh ; ry += stepSize )
          {
            for ( double rz = windowLow; rz <= windowHigh ; rz += stepSize )
            {
              std::vector<double> state;
              state.push_back(tx);
              state.push_back(ty);
              state.push_back(tz);
              state.push_back(rx);
              state.push_back(ry);
              state.push_back(rz);

              stateVector.push_back( state );
              for ( unsigned int trackerIndex = 0 ; trackerIndex < m_TrackingMatrixTimeStamps.size() ; trackerIndex++ )
              {
                std::vector <cv::Point3d> worldPoints;
                worldPoints.clear();
                for ( unsigned int frame = 0 ; frame < pointsInLensCS.size() ; frame++ )
                {
                  int framenumber = frame * 2;
                  worldPoints.push_back (GetCameraTrackingMatrix(framenumber, NULL , trackerIndex, &state ) *
                      pointsInLensCS[frame][0]);
                }
                
                cv::Point3d pointSpread;
                cv::Point3d worldCentre = mitk::GetCentroid (worldPoints, true,  &pointSpread);
                reconstructedPointSD[trackerIndex].push_back(pointSpread);
                std::vector <cv::Point3d > worldPoint(1);
                worldPoint[0] = worldCentre;
                projector->SetWorldPoints(worldPoint);
                projector->SetTrackerIndex(trackerIndex);
                projector->Project(this, &state);
                std::vector < std::vector < std::pair < cv::Point2d, cv::Point2d > > > projectedPoints = 
                projector->GetProjectedPoints();
                
                std::pair <double, double> projectedRMS = mitk::RMSError ( projectedPoints,  onScreenPoints );
                projectedErrorRMS[trackerIndex].push_back(projectedRMS);
                
              }
            }
          }
        }
      }
    }
  }

  if ( fileout.length() != 0 ) 
  {
    for ( unsigned int trackerIndex = 0 ; trackerIndex < m_TrackingMatrixTimeStamps.size() ; trackerIndex++ )
    {
      *fout[trackerIndex] << "#lag SDx SDy SDz RMSLeft RMSRight" << std::endl;
      for ( unsigned int i = 0 ; i  < stateVector.size() ; i ++ )
      {
        *fout[trackerIndex] << stateVector[i][0] << " " << 
        stateVector[i][1] << " " << 
        stateVector[i][2] << " " << 
        stateVector[i][3] << " " << 
        stateVector[i][4] << " " << 
        stateVector[i][5] << " : " << 
        reconstructedPointSD[trackerIndex][i].x << " " <<
        reconstructedPointSD[trackerIndex][i].y << " " <<
        reconstructedPointSD[trackerIndex][i].z << " " <<
        projectedErrorRMS[trackerIndex][i].first << " " <<
        projectedErrorRMS[trackerIndex][i].second << std::endl;
      }
      fout[trackerIndex]->close();
    }
  }

  for ( unsigned int trackerIndex = 0 ; trackerIndex < m_TrackingMatrixTimeStamps.size() ; trackerIndex++ )
  {
    std::pair < unsigned int , unsigned int > minIndexes;
    std::pair < double , double > minValues = mitk::FindMinimumValues ( projectedErrorRMS[trackerIndex], &minIndexes );
    MITK_INFO << "Tracker Index " << trackerIndex << " min left RMS " << minValues.first << " at " << (int)minIndexes.first + windowLow << " ms ";  ;
    MITK_INFO << "Tracker Index " << trackerIndex << " min right RMS " << minValues.second << " at " << (int)minIndexes.second + windowLow << " ms ";  ;
  }


}
} // namespace

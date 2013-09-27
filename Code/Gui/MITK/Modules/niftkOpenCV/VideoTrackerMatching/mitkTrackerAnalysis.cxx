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

  std::ofstream fout;
  if ( fileout.length() != 0 ) 
  {
    fout.open(fileout.c_str());
    if ( !fout )
    {
      MITK_WARN << "Failed to open output file for temporal calibration " << fileout;
    }
  }

  std::vector <cv::Point3d> pointsInLensCS;
  pointsInLensCS.clear();
  std::vector < std::pair < cv::Point2d, cv::Point2d > >* onScreenPoints = new std::vector < std::pair <cv::Point2d, cv::Point2d > >;
  onScreenPoints->clear();
  pointsInLensCS = ReadPointsInLensCSFile(calibrationfilename, onScreenPoints);

  if ( pointsInLensCS.size() * 2 != m_FrameNumbers.size() )
  {
    MITK_ERROR << "Temporal calibration file has wrong number of frames, " << pointsInLensCS.size() * 2 << " != " << m_FrameNumbers.size() ;
    return;
  }

  std::vector < std::vector <cv::Point3d> > standardDeviations;
  if ( fout ) 
  {
    fout << "#lag " ;
  }
  for ( unsigned int i = 0 ; i < m_TrackingMatrixTimeStamps.size() ; i++ )
  {
    std::vector <cv::Point3d> pointvector;
    standardDeviations.push_back(pointvector);
    if ( fout ) 
    {
      fout << "SDx SDy SDz";
    }
  }
  if ( fout ) 
  {
    fout << std::endl;
  }

  mitk::ProjectPointsOnStereoVideo::Pointer projector = mitk::ProjectPointsOnStereoVideo::New();
  projector->SetTrackerMatcher (this);
  projector->Initialise(m_Directory,m_CalibrationDirectory);

  std::vector < std::vector < cv::Point3d > > reconstructedPointSD;
  std::vector < std::vector < std::pair <double, double > > > projectedErrorRMS;
  for ( unsigned int i = 0 ; i < m_TrackingMatrixTimeStamps.size() ; i ++ ) 
  {
    std::vector < std::pair <double, double > > errors (windowHigh - windowLow);
    std::vector <cv::Point3d> pointerrors (windowHigh-windowLow);
    reconstructedPointSD.push_back(pointerrors);
    projectedErrorRMS.push_back(errors);
  }
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
   
    if ( fout ) 
    {
      fout << videoLag << " " ;
    }

    for ( unsigned int trackerIndex = 0 ; trackerIndex < m_TrackingMatrixTimeStamps.size() ; trackerIndex++ )
    {
      std::vector <cv::Point3d> worldPoints;
      worldPoints.clear();
      for ( unsigned int frame = 0 ; frame < pointsInLensCS.size() ; frame++ )
      {
        int framenumber = frame * 2;
        worldPoints.push_back (GetCameraTrackingMatrix(framenumber, NULL , trackerIndex ) *
            pointsInLensCS[frame]);
      }

      cv::Point3d worldCentre = mitk::GetCentroid (worldPoints, true, 
          &reconstructedPointSD[trackerIndex][videoLag - windowLow]);
      std::vector <cv::Point3d > worldPoint(1);
      worldPoint[0] = worldCentre;
      projector->SetWorldPoints(worldPoint);
      projector->Project();
      std::vector < std::vector < std::pair < cv::Point2d, cv::Point2d > > > projectedPoints = 
        projector->GetProjectedPoints();
      
      projectedErrorRMS[trackerIndex][videoLag - windowLow] = 
        mitk::RMSError ( projectedPoints[0],  *onScreenPoints ) ;
      
    }
  }

  MITK_INFO << "min sd at " ;
  //we've filled vectors with projection and recontruction errors, now we need to 
  //get the optimal values and output them

}
//---------------------------------------------------------------------------
void TrackerAnalysis::OptimiseHandeyeCalibration(std::string calibrationfilename ,
    bool visualise, std::string fileout)
{
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

  std::vector <cv::Point3d> pointsInLensCS;
  pointsInLensCS.clear();
  std::vector <std::pair <cv::Point2d, cv::Point2d > >* onScreenPoints = new std::vector <std::pair<cv::Point2d, cv::Point2d> >;
  onScreenPoints->clear();
  pointsInLensCS = ReadPointsInLensCSFile(calibrationfilename, onScreenPoints);

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
      if ( ! ( boost::math::isnan(pointsInLensCS[frame].x) || boost::math::isnan(pointsInLensCS[frame].y) || boost::math::isnan(pointsInLensCS[frame].z) ) )
      {
        cameraMatrices.push_back (GetCameraTrackingMatrix(framenumber, NULL , trackerIndex ));
      }
      else
      {
       pointsInLensCS.erase(pointsInLensCS.begin() + frame);
       frame -- ;
      }
    }
    bool optimiseScaling = false;
    bool optimiseInvariantPoint = true;
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
    invPointCal->Calibrate(cameraMatrices, pointsInLensCS,
        optimiseScaling, optimiseInvariantPoint, rigidBodyTransformation,
        invariantPoint, millimetresPerPixel,
        outputMatrix, residualError);
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
    bool visualise, std::string fileout)
{
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

  std::vector <cv::Point3d> pointsInLensCS;
  pointsInLensCS.clear();
  std::vector <std::pair <cv::Point2d, cv::Point2d > >* onScreenPoints = new std::vector <std::pair<cv::Point2d, cv::Point2d> >;
  onScreenPoints->clear();
  pointsInLensCS = ReadPointsInLensCSFile(calibrationfilename, onScreenPoints);

  if ( pointsInLensCS.size() * 2 != m_FrameNumbers.size() )
  {
    MITK_ERROR << "Temporal calibration file has wrong number of frames, " << pointsInLensCS.size() * 2 << " != " << m_FrameNumbers.size() ;
    return;
  }
}
} // namespace

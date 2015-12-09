/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include "mitkVideoTrackerMatching.h"
#include <mitkCameraCalibrationFacade.h>
#include <mitkOpenCVMaths.h>
#include <mitkOpenCVFileIOUtils.h>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <sstream>
#include <fstream>
#include <cstdlib>

namespace mitk 
{

//---------------------------------------------------------------------------
VideoTrackerMatching::VideoTrackerMatching () 
: m_Ready(false)
, m_FlipMatrices(false)
, m_WriteTimingErrors(false)
, m_HaltOnFrameSkip(true)
, m_InterpolateMatrices(false)
{}


//---------------------------------------------------------------------------
VideoTrackerMatching::~VideoTrackerMatching () 
{}


//---------------------------------------------------------------------------
void VideoTrackerMatching::Initialise(std::string directory)
{
  m_Directory = directory;
  std::vector<std::string> FrameMaps = mitk::FindVideoFrameMapFiles(m_Directory);

  if ( FrameMaps.size() != 1 ) 
  {
    MITK_ERROR << "Found " << FrameMaps.size() << " framemap.log files, VideoTrackerMatching failed to initialise.";
    m_Ready=false;
    return;
  }
  else
  {
    MITK_INFO << "Found " << FrameMaps[0];
    m_TrackingMatrixDirectories = mitk::FindTrackingMatrixDirectories(m_Directory);
    if ( m_TrackingMatrixDirectories.size() == 0 ) 
    {
      MITK_ERROR << "Found no tracking directories, VideoTrackerMatching failed to initiliase.";
      m_Ready=false;
      return;
    }
    else 
    {
      for ( unsigned int i = 0 ; i < m_TrackingMatrixDirectories.size() ; i ++ ) 
      {
        TrackingAndTimeStampsContainer tempTimeStamps = mitk::TrackingAndTimeStampsContainer();
        tempTimeStamps.LoadFromDirectory(m_TrackingMatrixDirectories[i]);
        MITK_INFO << "Found " << tempTimeStamps.GetSize() << " time stamped tracking files in " << m_TrackingMatrixDirectories[i];
        m_TrackingMatricesAndTimeStamps.push_back(tempTimeStamps);
        m_VideoLag.push_back(0);
        m_VideoLeadsTracking.push_back(false);
        std::vector < long long > emptyErrors;
        m_TimingErrors.push_back(emptyErrors);
        cv::Mat tempCameraToTracker = cv::Mat(4,4,CV_64F);
        for ( int i = 0 ; i < 4 ; i ++ ) 
        {
          for ( int j = 0 ; j < 4 ; j ++ )
          {
            if ( i == j ) 
            {
              tempCameraToTracker.at<double>(i,j) = 1.0;
            }
            else
            {
              tempCameraToTracker.at<double>(i,j) = 0.0;
            }
          }
        }
        m_CameraToTracker.push_back(tempCameraToTracker);
      }
    }
  }
  m_FrameMap = FrameMaps[0];
  ProcessFrameMapFile();
  if ( CheckTimingErrorStats() )
  { 
    MITK_INFO << "VideoTrackerMatching initialised OK";
    m_Ready=true;
  }
  else
  {
    MITK_WARN << "VideoTrackerMatching initialise FAILED";
    m_Ready=false;
  }
  return;
}

//---------------------------------------------------------------------------
void VideoTrackerMatching::ProcessFrameMapFile ()
{
  std::ifstream fin(m_FrameMap.c_str());
  if ( !fin )
  {
    MITK_WARN << "Failed to open frame map file " << m_FrameMap;
    return;
  }

  std::string line;
  unsigned int frameNumber; 
  unsigned int sequenceNumber;
  unsigned int channel;
  unsigned long long timeStamp;
  unsigned int linenumber = 0;
  cv::Mat trackingMatrix ( 4, 4, CV_64FC1 );

  m_FrameNumbers.clear();
  for ( unsigned int i = 0 ; i < m_TrackingMatricesAndTimeStamps.size() ; i ++ )
  {
    m_TimingErrors[i].clear();
  }
  unsigned int badMatrices = 0;
  while ( getline(fin,line) )
  {
    if ( line[0] != '#' )
    {
      std::stringstream linestream(line);
      linestream >> frameNumber >> sequenceNumber >> channel >> timeStamp;
       if ( linestream.good() )
      {
        m_FrameNumbers.push_back(frameNumber);
        m_VideoTimeStamps.Insert(timeStamp);
        
        for ( unsigned int i = 0 ; i < m_TrackingMatricesAndTimeStamps.size() ; i ++ )
        {
          long long timingError;
          if ( m_VideoLeadsTracking[i] )
          {
            m_TrackingMatricesAndTimeStamps[i].GetNearestTimeStamp(
                timeStamp + m_VideoLag[i], &timingError);
          }
          else
          {
            m_TrackingMatricesAndTimeStamps[i].GetNearestTimeStamp(
                timeStamp - m_VideoLag[i], &timingError);
          }
          
          m_TimingErrors[i].push_back(timingError);
        }

        if ( frameNumber != linenumber++ )
        {
          MITK_WARN << "Skipped frame detected at line " << linenumber ;
          if ( m_HaltOnFrameSkip ) 
          {
            MITK_ERROR << "Halt on frame skip true, so halting, check data";
            exit(1);
          }
        }
      }
      else
      {
        MITK_WARN << "Parse failure at line " << linenumber;
      }
    }
  }
    
  MITK_INFO << "Read " << linenumber << " lines from " << m_FrameMap;
  MITK_INFO << "Got " << badMatrices << " bad matrix files ";
}


//---------------------------------------------------------------------------
void VideoTrackerMatching::SetVideoLagMilliseconds ( unsigned long long VideoLag, bool VideoLeadsTracking , int trackerIndex) 
{
  if ( trackerIndex == -1 ) 
  {
    for ( unsigned int i = 0 ; i < m_VideoLag.size() ; i ++ )
    { 
      m_VideoLag[i] = VideoLag * 1e6;
      m_VideoLeadsTracking[i] = VideoLeadsTracking;
    }
  }
  else
  {
    m_VideoLag[trackerIndex] = VideoLag * 1e6;
    m_VideoLeadsTracking[trackerIndex] = VideoLeadsTracking;
  }

  if ( m_Ready ) 
  {
    MITK_INFO << "Set video lag after initialisation reprocessing frame map files";
    ProcessFrameMapFile();
    if ( CheckTimingErrorStats() )
    { 
      MITK_INFO << "VideoTrackerMatching initialised OK";
      m_Ready=true;
    }
    else
    {
      MITK_WARN << "VideoTrackerMatching initialise FAILED";
      m_Ready=false;
    }
  }
  return;
}


//---------------------------------------------------------------------------
bool VideoTrackerMatching::CheckTimingErrorStats()
{
  bool ok = true;
  //check sizes
  if ( m_TrackingMatricesAndTimeStamps.size() != m_TrackingMatrixDirectories.size() )
  {
    MITK_ERROR << "Wrong number of tracking matrix dirtectories " << m_TrackingMatricesAndTimeStamps.size() 
      << " != " <<  m_TrackingMatrixDirectories.size();
    ok=false;
  }
  for ( unsigned int i = 0 ; i < m_TrackingMatricesAndTimeStamps.size() ; i ++ ) 
  {
    if ( m_TimingErrors[i].size() != m_FrameNumbers.size() )
    {
      MITK_ERROR << "Wrong number of frame numbers " << i << ": " << m_TimingErrors[i].size()
        << " != " <<  m_FrameNumbers.size();
      ok = false;
    }
  }

  for ( unsigned int i = 0 ; i < m_TrackingMatricesAndTimeStamps.size() ; i++ )
  {
    std::ofstream fout;
    if ( m_WriteTimingErrors )
    {
       std::string fileout = m_TrackingMatrixDirectories[i] + ".timimgErrors";
       fout.open(fileout.c_str());
       if ( fout )
       {
         MITK_INFO << "Writing Timing errors to " << fileout;
       }
       else
       {
         MITK_ERROR << "Failed to open " << fileout << " to write timing errors to";
       }
    }
    
    double mean = 0 ; 
    double absmean = 0 ; 
    long long minimum = m_TimingErrors[i][0];
    long long maximum = m_TimingErrors[i][0];

    for (unsigned int j = 0 ; j < m_TimingErrors[i].size() ; j ++ ) 
    {
      mean += static_cast<double>(m_TimingErrors[i][j]);
      absmean += fabs(static_cast<double>(m_TimingErrors[i][j]));
      minimum = m_TimingErrors[i][j] < minimum ? m_TimingErrors[i][j] : minimum;
      maximum = m_TimingErrors[i][j] > maximum ? m_TimingErrors[i][j] : maximum;
      if ( fout ) 
      {
        fout << static_cast<double>(m_TimingErrors[i][j]) << std::endl;
      }
    }
    mean /= m_TimingErrors[i].size();
    absmean /= m_TimingErrors[i].size();
    
    MITK_INFO << "There are " << m_TimingErrors[i].size() << " matched frames in data set " << i;
    MITK_INFO << "Average timing error for set " << i << " = " << mean * 1e-6 << "ms";
    MITK_INFO << "Average absolute timing error for set " << i << " = " << absmean * 1e-6 << "ms";
    MITK_INFO << "Maximum timing error for set " << i << " = " << maximum * 1e-6 << "ms";
    MITK_INFO << "Minimum timing error for set " << i << " = " << minimum * 1e-6 << "ms";
  }

  return ok;
}


//---------------------------------------------------------------------------
void VideoTrackerMatching::SetCameraToTracker (cv::Mat matrix, int trackerIndex)
{
  if ( ! m_Ready )
  {
    MITK_ERROR << "Need to initialise tracker matcher before setting camera to tracker matrices";
    return;
  }
  if ( trackerIndex == -1 )
  {
    for ( unsigned int i = 0 ; i < m_CameraToTracker.size() ; i ++ )
    {
      m_CameraToTracker[i] = matrix;
    }
  }
  else
  {
    m_CameraToTracker[trackerIndex] = matrix;
  }

}


//---------------------------------------------------------------------------
cv::Mat VideoTrackerMatching::GetTrackerMatrix ( unsigned int FrameNumber , long long * TimingError  ,unsigned int TrackerIndex,
    int ReferenceIndex )
{
  cv::Mat returnMat = cv::Mat(4,4,CV_64FC1);
  cv::Mat referenceMat = cv::Mat(4,4,CV_64FC1);
  
  if ( !m_Ready ) 
  {
    MITK_WARN << "Attempted to get tracking matrix when videoTrackerMatching not initialised.";
    return returnMat;
  }

  if ( TrackerIndex >= m_TrackingMatricesAndTimeStamps.size () )
  {
    MITK_WARN << "Attempted to get tracking matrix with invalid TrackerIndex";
    return returnMat;
  }

  if ( FrameNumber >= m_VideoTimeStamps.GetSize() )
  {
    MITK_WARN << "Attempted to get tracking matrix with invalid frame index";
    return returnMat;
  }

  mitk::TimeStampsContainer::TimeStamp targetTimeStamp = m_VideoTimeStamps.GetTimeStamp(FrameNumber); 
  if ( m_VideoLeadsTracking[TrackerIndex] )
  {
    targetTimeStamp += m_VideoLag[TrackerIndex];
  }
  else
  {
    targetTimeStamp -= m_VideoLag[TrackerIndex];
  }
         
  //now we need to get nearest tracker matrix, either by interpolation or not.
  long long timingError;
  bool inBounds;
  
  if ( ! m_InterpolateMatrices )
  {
    returnMat=cv::Mat(m_TrackingMatricesAndTimeStamps[TrackerIndex].GetNearestMatrix(targetTimeStamp, timingError, inBounds));
  }
  else
  {
    returnMat=cv::Mat(m_TrackingMatricesAndTimeStamps[TrackerIndex].InterpolateMatrix(targetTimeStamp, timingError, inBounds));
  }

  if ( ReferenceIndex != -1 )
  {
    long long refTimingError;
    bool refInBounds;
    
    mitk::TimeStampsContainer::TimeStamp refTargetTimeStamp = m_VideoTimeStamps.GetTimeStamp(FrameNumber); 
    if ( m_VideoLeadsTracking[ReferenceIndex] )
    {
      refTargetTimeStamp += m_VideoLag[ReferenceIndex];
    }
    else
    {
      refTargetTimeStamp -= m_VideoLag[ReferenceIndex];
    }
  
    if ( ! m_InterpolateMatrices )
    {
      referenceMat=cv::Mat(m_TrackingMatricesAndTimeStamps[ReferenceIndex].GetNearestMatrix(refTargetTimeStamp, refTimingError, refInBounds));
    }
    else
    {
      referenceMat=cv::Mat(m_TrackingMatricesAndTimeStamps[ReferenceIndex].InterpolateMatrix(refTargetTimeStamp, refTimingError, refInBounds));
    }
    returnMat = referenceMat.inv() * returnMat; 
    if ( std::abs ( refTimingError ) > std::abs ( timingError ) )
    {
      timingError = refTimingError;
    }
  }

  if ( TimingError != NULL ) 
  {
    *TimingError = timingError;
  }
  
  if ( m_FlipMatrices )
  {
    //flip the matrix between left and right handed coordinate systems
    std::vector<cv::Mat> theseMats;
    theseMats.push_back(returnMat);
    std::vector<cv::Mat> flippedMats = mitk::FlipMatrices(theseMats);
    return flippedMats[0];
  }
  else
  {
    return returnMat;
  }
}

//---------------------------------------------------------------------------
cv::Mat VideoTrackerMatching::GetVideoFrame ( unsigned int frameNumber , unsigned long long * timeStamp )
{
  //a dummy holder for the return matrix, This should be implemented properly
  cv::Mat returnMat = cv::Mat(4,4,CV_64FC1);
  if ( timeStamp != NULL )
  {
    *timeStamp = m_VideoTimeStamps.GetTimeStamp(frameNumber);
  }
  return returnMat;
}

//---------------------------------------------------------------------------
unsigned long long VideoTrackerMatching::GetVideoFrameTimeStamp ( unsigned int frameNumber )
{
  return m_VideoTimeStamps.GetTimeStamp(frameNumber);
}


//---------------------------------------------------------------------------
cv::Mat VideoTrackerMatching::GetCameraTrackingMatrix ( unsigned int FrameNumber , long long * TimingError  ,unsigned int TrackerIndex  , std::vector <double>* Perturbation, int ReferenceIndex )
{
   cv::Mat trackerMatrix = GetTrackerMatrix ( FrameNumber, TimingError, TrackerIndex );
   if ( Perturbation == NULL ) 
   {
     if ( ReferenceIndex == -1 ) 
     {
       return trackerMatrix * m_CameraToTracker[TrackerIndex];
     }
     else
     {
       long long  refTimingError;
       cv::Mat referenceMatrix = GetTrackerMatrix ( FrameNumber, &refTimingError, ReferenceIndex );
       cv::Mat toReference = referenceMatrix * m_CameraToTracker[ReferenceIndex];
       if ( TimingError != NULL && refTimingError > *TimingError )
       {  
         *TimingError = refTimingError;
       }
       return (toReference.inv() * (trackerMatrix * m_CameraToTracker[TrackerIndex]));
     }

   }
   else
   {
     assert ( Perturbation->size() == 6 );
     if ( ReferenceIndex == -1 )
     {
        return trackerMatrix * 
          mitk::PerturbTransform ( m_CameraToTracker[TrackerIndex],
            Perturbation->at(0), Perturbation->at(1), Perturbation->at(2), 
            Perturbation->at(3), Perturbation->at(4), Perturbation->at(5));
     }
     else
     {
       long long  refTimingError;
       cv::Mat referenceMatrix = GetTrackerMatrix ( FrameNumber, &refTimingError, ReferenceIndex );
       cv::Mat toReference = referenceMatrix * m_CameraToTracker[ReferenceIndex];
       if ( TimingError != NULL && refTimingError > *TimingError )
       {  
         *TimingError = refTimingError;
       }
       return toReference.inv() * 
          (trackerMatrix *  
          mitk::PerturbTransform ( m_CameraToTracker[TrackerIndex],
            Perturbation->at(0), Perturbation->at(1), Perturbation->at(2), 
            Perturbation->at(3), Perturbation->at(4), Perturbation->at(5)) );
     }
   }
}


//---------------------------------------------------------------------------
void VideoTrackerMatching::SetCameraToTrackers(std::string filename)
{
  if ( !m_Ready )
  {
    MITK_ERROR << "Initialise video tracker matcher before setting camera to trackers.";
    return;
  }
  std::ifstream fin(filename.c_str());
  if ( !fin )
  {
    MITK_WARN << "Failed to open camera to tracker file " << filename;
    return;
  }

  std::string line;
  unsigned int indexnumber = 0;

  int row = 0 ;
  while ( getline(fin,line) )
  {
    if ( line[0] != '#' )
    {
      std::stringstream linestream(line);
      linestream >> m_CameraToTracker[indexnumber].at<double>(row,0) >>
        m_CameraToTracker[indexnumber].at<double>(row,1) >>
        m_CameraToTracker[indexnumber].at<double>(row,2) >>
        m_CameraToTracker[indexnumber].at<double>(row,3);

      if ( ! linestream.fail() )
      {
        row++;
        if ( row == 4 ) 
        {
          row = 0 ; 
          indexnumber++;
        }
      } 
      else
      {
        MITK_WARN << "Parse failure at line ";
      }
    }
  }
  fin.close();
  MITK_INFO << "Read handeye's from " << filename;
  for ( unsigned int i = 0 ; i < m_CameraToTracker.size() ; i ++ )
  {
    MITK_INFO << m_CameraToTracker[i];
  }
} 


//---------------------------------------------------------------------------
std::vector < mitk::WorldPointsWithTimingError > VideoTrackerMatching::ReadPointsInLensCSFile
(std::string calibrationfilename, 
    int PointsPerFrame, 
    std::vector < mitk::ProjectedPointPairsWithTimingError > * onScreenPoints )
{
  std::vector < mitk::WorldPointsWithTimingError >  pointsInLensCS;
  pointsInLensCS.clear();
  std::ifstream fin(calibrationfilename.c_str());
  if ( !fin )
  {
    MITK_WARN << "Failed to open points in lens CS file " << calibrationfilename;
    return pointsInLensCS;
  }
  std::string line;
  unsigned int frameNumber; 
  unsigned int linenumber = 0;
  
  std::getline (fin,line); 
  while ( fin.good() )
  {
    mitk::WorldPointsWithTimingError   framePointsInLensCS;
    mitk::ProjectedPointPairsWithTimingError frameOnScreenPoints;
    framePointsInLensCS.m_Points.clear();
    frameOnScreenPoints.m_Points.clear();
    for ( int pointID = 0 ; pointID < PointsPerFrame ; pointID ++ ) 
    {
      if ( line[0] != '#' )
      {
        std::stringstream linestream(line);
        std::string xstring;
        std::string ystring;
        std::string zstring;
        std::string lxstring;
        std::string lystring;
        std::string rxstring;
        std::string rystring;

        linestream >> frameNumber >> xstring >> ystring >> zstring
         >> lxstring >> lystring >> rxstring >> rystring;
        if ( linestream.good() )
        {
          framePointsInLensCS.m_Points.push_back(
              mitk::WorldPoint (cv::Point3d(
              atof (xstring.c_str()), atof(ystring.c_str()),atof(zstring.c_str()))) );  
          if ( onScreenPoints != NULL ) 
          {
            frameOnScreenPoints.m_Points.push_back(mitk::ProjectedPointPair (
                cv::Point2d(atof (lxstring.c_str()), atof ( lystring.c_str()) ) ,
                cv::Point2d(atof (rxstring.c_str()), atof ( rystring.c_str()) )));
          }
          if ( frameNumber != linenumber++ )
          {
            MITK_WARN << "Skipped frame detected at line " << linenumber ;
          }
        } 
        else
        {
          MITK_WARN << "Parse failure at line " << linenumber;
        }
      }
      else
      {
        pointID --;
      }
      std::getline (fin, line);
    }
    pointsInLensCS.push_back(framePointsInLensCS);
    if ( onScreenPoints != NULL )
    {
      onScreenPoints->push_back(frameOnScreenPoints);
    }
  }
  fin.close();
  return pointsInLensCS;
}

//---------------------------------------------------------------------------
unsigned int VideoTrackerMatching::GetTrackingMatricesSize ()
{
  return m_TrackingMatricesAndTimeStamps.size();
}

} // namespace

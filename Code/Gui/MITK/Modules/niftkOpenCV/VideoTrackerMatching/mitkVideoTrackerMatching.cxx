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
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
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
{}

//---------------------------------------------------------------------------
VideoTrackerMatching::~VideoTrackerMatching () 
{}

//---------------------------------------------------------------------------
void VideoTrackerMatching::Initialise(std::string directory)
{
  m_Directory = directory;
  std::vector<std::string> FrameMaps = FindFrameMaps();
  
  if ( FrameMaps.size() != 1 ) 
  {
    MITK_ERROR << "Found " << FrameMaps.size() << " framemap.log files, VideoTrackerMatching failed to initialise.";
    m_Ready=false;
    return;
  }
  else
  {
    MITK_INFO << "Found " << FrameMaps[0];
    FindTrackingMatrixDirectories();
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
        TrackingMatrixTimeStamps tempTimeStamps = FindTrackingTimeStamps(m_TrackingMatrixDirectories[i]);
        MITK_INFO << "Found " << tempTimeStamps.m_TimeStamps.size() << " time stamped tracking files in " << m_TrackingMatrixDirectories[i];
        m_TrackingMatrixTimeStamps.push_back(tempTimeStamps);
        m_VideoLag.push_back(0);
        m_VideoLeadsTracking.push_back(false);
        cv::Mat tempCameraToTracker = cv::Mat(4,4,CV_32F);
        for ( int i = 0 ; i < 4 ; i ++ ) 
        {
          for ( int j = 0 ; j < 4 ; j ++ )
          {
            if ( i == j ) 
            {
              tempCameraToTracker.at<float>(i,j) = 1.0;
            }
            else
            {
              tempCameraToTracker.at<float>(i,j) = 0.0;
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
std::vector<std::string> VideoTrackerMatching::FindFrameMaps()
{
  boost::filesystem::recursive_directory_iterator end_itr;
  boost::regex framelogfilter ( "(.+)(framemap.log)");
  std::vector<std::string> returnStrings;

  for ( boost::filesystem::recursive_directory_iterator it(m_Directory); 
      it != end_itr ; ++it)
   {
     if ( boost::filesystem::is_regular_file (it->status()) )
     {
       boost::cmatch what;
       //  if ( it->path().extension() == ".framemap.log" )
       const std::string stringthing = it->path().filename().string();

       if ( boost::regex_match( stringthing.c_str(), what, framelogfilter) )
       {
         returnStrings.push_back(it->path().string());
       }
     }
   }
  return returnStrings;
}

//---------------------------------------------------------------------------
void VideoTrackerMatching::FindTrackingMatrixDirectories()
{
  //need to work in this
  boost::filesystem::recursive_directory_iterator end_itr;
  for ( boost::filesystem::recursive_directory_iterator it(m_Directory); 
      it != end_itr ; ++it)
   {
     if ( boost::filesystem::is_directory (it->status()) )
     {
       if ( CheckIfDirectoryContainsTrackingMatrices(it->path().string()))
       {
          m_TrackingMatrixDirectories.push_back(it->path().string());
          //need to init tracking matrix vector
          TrackingMatrices TempMatrices; 
          m_TrackingMatrices.push_back(TempMatrices);
       }
     }
   }
  return;
}
//---------------------------------------------------------------------------
TrackingMatrixTimeStamps VideoTrackerMatching::FindTrackingTimeStamps(std::string directory)
{
  boost::filesystem::directory_iterator end_itr;
  boost::regex TimeStampFilter ( "([0-9]{19})(.txt)");
  TrackingMatrixTimeStamps ReturnStamps;
  for ( boost::filesystem::directory_iterator it(directory);it != end_itr ; ++it)
   {
   if ( boost::filesystem::is_regular_file (it->status()) )
    {
      boost::cmatch what;
      std::string stringthing = it->path().filename().string();
      if ( boost::regex_match( stringthing.c_str(),what , TimeStampFilter) )
      {
        ReturnStamps.m_TimeStamps.push_back(boost::lexical_cast<unsigned long long>(it->path().filename().stem().string().c_str()));
      }
    }
  }
  //sort the vectorreinterpret_cast<const char*>
  std::sort ( ReturnStamps.m_TimeStamps.begin() , ReturnStamps.m_TimeStamps.end());
  return ReturnStamps;
}
//---------------------------------------------------------------------------
bool VideoTrackerMatching::CheckIfDirectoryContainsTrackingMatrices(std::string directory)
{
  boost::filesystem::directory_iterator end_itr;
  boost::regex TimeStampFilter ( "([0-9]{19})(.txt)");
  for ( boost::filesystem::directory_iterator it(directory);it != end_itr ; ++it)
   {
   if ( boost::filesystem::is_regular_file (it->status()) )
    {
      boost::cmatch what;
      std::string stringthing = it->path().filename().string();
      if ( boost::regex_match( stringthing.c_str(),what , TimeStampFilter) )
      {
        return true;
      }
    }
  }
  return false;
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
  unsigned int SequenceNumber;
  unsigned int channel;
  unsigned long long TimeStamp;
  unsigned int linenumber = 0;

  m_FrameNumbers.clear();
  for ( unsigned int i = 0 ; i < m_TrackingMatrixTimeStamps.size() ; i ++ )
  {
    m_TrackingMatrices[i].m_TimingErrors.clear();
    m_TrackingMatrices[i].m_TrackingMatrices.clear();
  }
  while ( getline(fin,line) )
  {
    if ( line[0] != '#' )
    {
      std::stringstream linestream(line);
      bool parseSuccess = linestream >> frameNumber >> SequenceNumber >> channel >> TimeStamp;
       if ( parseSuccess )
      {
        m_FrameNumbers.push_back(frameNumber);
        for ( unsigned int i = 0 ; i < m_TrackingMatrixTimeStamps.size() ; i ++ )
        {
          long long * timingError = new long long;
          unsigned long long TargetTimeStamp; 
          if ( m_VideoLeadsTracking[i] )
          {
            TargetTimeStamp = m_TrackingMatrixTimeStamps[i].GetNearestTimeStamp(
                TimeStamp + m_VideoLag[i],timingError);
          }
          else
          {
            TargetTimeStamp = m_TrackingMatrixTimeStamps[i].GetNearestTimeStamp(
                TimeStamp - m_VideoLag[i],timingError);
          }
          
          m_TrackingMatrices[i].m_TimingErrors.push_back(*timingError);

          std::string MatrixFileName = boost::lexical_cast<std::string>(TargetTimeStamp) + ".txt";
          boost::filesystem::path MatrixFileNameFull (m_TrackingMatrixDirectories[i]);
          MatrixFileNameFull /= MatrixFileName;

          m_TrackingMatrices[i].m_TrackingMatrices.push_back(ReadTrackerMatrix(MatrixFileNameFull.string()));

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
  }
  MITK_INFO << "Read " << linenumber << " lines from " << m_FrameMap;
    
}

//---------------------------------------------------------------------------
unsigned long long TrackingMatrixTimeStamps::GetNearestTimeStamp (unsigned long long timestamp, long long * Delta)
{
  std::vector<unsigned long long>::iterator upper = std::upper_bound (m_TimeStamps.begin() , m_TimeStamps.end(), timestamp);
  std::vector<unsigned long long>::iterator lower = std::lower_bound (m_TimeStamps.begin() , m_TimeStamps.end(), timestamp);
  long long deltaUpper = *upper - timestamp ;
  long long deltaLower = timestamp - *lower ;
  unsigned long long returnValue;
  long long delta;
  if ( deltaLower == 0 ) 
  {
    returnValue = *lower;
    delta = 0;
  }
  else
  {
    deltaLower = timestamp - *(--lower);
    if ( abs(deltaLower) < abs(deltaUpper) ) 
    {
      returnValue = *lower;
      delta = timestamp - *lower;
    }
    else
    {
      returnValue = *upper;
      delta = timestamp - *upper;
    }
  }

  if ( Delta != NULL ) 
  {
    *Delta = delta;
  }
  return returnValue;
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
cv::Mat VideoTrackerMatching::ReadTrackerMatrix(std::string filename)
{
  cv::Mat TrackerMatrix = cv::Mat(4,4, CV_32FC1);
  std::ifstream fin(filename.c_str());
  if ( !fin )
  {
    MITK_WARN << "Failed to open matrix file " << filename;
    return TrackerMatrix;
  }
  for ( int row = 0 ; row < 4 ; row ++ )
  {
    for ( int col = 0 ; col < 4 ; col ++ ) 
    {
      fin >> TrackerMatrix.at<float>(row,col);
    }
  }
  return TrackerMatrix;
}
//---------------------------------------------------------------------------
bool VideoTrackerMatching::CheckTimingErrorStats()
{
  bool ok = true;
  //check sizes
  if ( m_TrackingMatrices.size() != m_TrackingMatrixDirectories.size() )
  {
    MITK_ERROR << "Wrong number of tracking matrix dirtectories " << m_TrackingMatrices.size() 
      << " != " <<  m_TrackingMatrixDirectories.size();
    ok=false;
  }
  for ( unsigned int i = 0 ; i < m_TrackingMatrices.size() ; i ++ ) 
  {
    if ( m_TrackingMatrices[i].m_TrackingMatrices.size() != 
        m_TrackingMatrices[i].m_TimingErrors.size() )
    {
      MITK_ERROR << "Wrong number of tracking matrices " << i << ": " << m_TrackingMatrices[i].m_TrackingMatrices.size() 
        << " != " <<  m_TrackingMatrices[i].m_TimingErrors.size();
      ok = false;
    }
    if ( m_TrackingMatrices[i].m_TrackingMatrices.size() != 
        m_FrameNumbers.size() )
    {
      MITK_ERROR << "Wrong number of frame numbers " << i << ": " << m_TrackingMatrices[i].m_TrackingMatrices.size() 
        << " != " <<  m_FrameNumbers.size();
      ok = false;
    }
  }

  for ( unsigned int i = 0 ; i < m_TrackingMatrices.size() ; i++ )
  {
    double mean = 0 ; 
    double absmean = 0 ; 
    long long minimum = m_TrackingMatrices[i].m_TimingErrors[0];
    long long maximum = m_TrackingMatrices[i].m_TimingErrors[0];

    for (unsigned int j = 0 ; j < m_TrackingMatrices[i].m_TimingErrors.size() ; j ++ ) 
    {
      mean += static_cast<double>(m_TrackingMatrices[i].m_TimingErrors[j]);
      absmean += fabs(static_cast<double>(m_TrackingMatrices[i].m_TimingErrors[j]));
      minimum = m_TrackingMatrices[i].m_TimingErrors[j] < minimum ? m_TrackingMatrices[i].m_TimingErrors[j] : minimum;
      maximum = m_TrackingMatrices[i].m_TimingErrors[j] > maximum ? m_TrackingMatrices[i].m_TimingErrors[j] : maximum;

    }
    mean /= m_TrackingMatrices[i].m_TimingErrors.size();
    absmean /= m_TrackingMatrices[i].m_TimingErrors.size();
    
    MITK_INFO << "There are " << m_TrackingMatrices[i].m_TimingErrors.size() << " matched frames in data set " << i;
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
cv::Mat VideoTrackerMatching::GetTrackerMatrix ( unsigned int FrameNumber , long long * TimingError  ,unsigned int TrackerIndex  )
{
  cv::Mat returnMat = cv::Mat(4,4,CV_32FC1);
  
  if ( !m_Ready ) 
  {
    MITK_WARN << "Attempted to get tracking matrix when videoTrackerMatching not initialised.";
    return returnMat;
  }

  if ( TrackerIndex >= m_TrackingMatrices.size () )
  {
    MITK_WARN << "Attempted to get tracking matrix with invalid TrackerIndex";
    return returnMat;
  }

  if ( FrameNumber >= m_TrackingMatrices[TrackerIndex].m_TrackingMatrices.size() )
  {
    MITK_WARN << "Attempted to get tracking matrix with invalid frame index";
    return returnMat;
  }

  returnMat=m_TrackingMatrices[TrackerIndex].m_TrackingMatrices[FrameNumber];
  if ( TimingError != NULL ) 
  {
    *TimingError = m_TrackingMatrices[TrackerIndex].m_TimingErrors[FrameNumber];
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
cv::Mat VideoTrackerMatching::GetCameraTrackingMatrix ( unsigned int FrameNumber , long long * TimingError  ,unsigned int TrackerIndex  )
{
   cv::Mat TrackerMatrix = GetTrackerMatrix ( FrameNumber, TimingError, TrackerIndex );
   return TrackerMatrix * m_CameraToTracker[TrackerIndex];
}
//---------------------------------------------------------------------------
void VideoTrackerMatching::TemporalCalibration(std::string calibrationfilename ,
    int windowLow, int windowHigh, bool visualise, std::string fileout)
{
  if ( !m_Ready )
  {
    MITK_ERROR << "Initialise video tracker matcher before attempting temporal calibration";
    return;
  }
  std::ifstream fin(calibrationfilename.c_str());
  if ( !fin )
  {
    MITK_WARN << "Failed to open temporal calibration file " << calibrationfilename;
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

  std::string line;
  unsigned int frameNumber; 
  unsigned int linenumber = 0;
  std::vector <cv::Point3f> pointsInLensCS;
  pointsInLensCS.clear();

  while ( getline(fin,line) )
  {
    if ( line[0] != '#' )
    {
      std::stringstream linestream(line);
      std::string xstring;
      std::string ystring;
      std::string zstring;

      bool parseSuccess = linestream >> frameNumber >> xstring >> ystring >> zstring;
      if ( parseSuccess )
      {
        pointsInLensCS.push_back(cv::Point3f(
              atof (xstring.c_str()), atof(ystring.c_str()),atof(zstring.c_str())));  
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
  }
  fin.close();
  if ( pointsInLensCS.size() * 2 != m_FrameNumbers.size() )
  {
    MITK_ERROR << "Temporal calibration file has wrong number of frames, " << pointsInLensCS.size() * 2 << " != " << m_FrameNumbers.size() ;
    return;
  }

  std::vector < std::vector <cv::Point3f> > standardDeviations;
  if ( fout ) 
  {
    fout << "#lag " ;
  }
  for ( unsigned int i = 0 ; i < m_TrackingMatrixTimeStamps.size() ; i++ )
  {
    std::vector <cv::Point3f> pointvector;
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
      std::vector <cv::Point3f> worldPoints;
      worldPoints.clear();
      for ( unsigned int frame = 0 ; frame < pointsInLensCS.size() ; frame++ )
      {
        int framenumber = frame * 2;
        worldPoints.push_back (GetCameraTrackingMatrix(framenumber, NULL , trackerIndex ) *
            pointsInLensCS[frame]);
      }
      cv::Point3f* worldStdDev = new cv::Point3f;
      mitk::GetCentroid (worldPoints, true, worldStdDev);
      standardDeviations[trackerIndex].push_back(*worldStdDev);
      if ( fout ) 
      {
        fout << *worldStdDev << " ";
      }

    }
    if ( fout ) 
    {
      fout << std::endl;
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
      bool parseSuccess = linestream >> m_CameraToTracker[indexnumber].at<float>(row,0) >>
        m_CameraToTracker[indexnumber].at<float>(row,1) >>
        m_CameraToTracker[indexnumber].at<float>(row,2) >>
        m_CameraToTracker[indexnumber].at<float>(row,3);

      if ( parseSuccess )
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
} // namespace

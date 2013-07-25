/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkVideoTrackerMatching.h"
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>

#include <sstream>
#include <fstream>
namespace mitk 
{
VideoTrackerMatching::VideoTrackerMatching () 
: m_Ready(false)
{}

VideoTrackerMatching::~VideoTrackerMatching () 
{}

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
      }
    }
  }

  ProcessFrameMapFile(FrameMaps[0]);
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
std::vector<std::string> VideoTrackerMatching::FindFrameMaps()
{
  boost::filesystem::recursive_directory_iterator end_itr;
  boost::regex framelogfilter ( "(.+)(framemap.log)");
  std::vector<std::string> ReturnStrings;
  for ( boost::filesystem::recursive_directory_iterator it(m_Directory); 
      it != end_itr ; ++it)
   {
     if ( boost::filesystem::is_regular_file (it->status()) )
     {
       boost::cmatch what;
      //  if ( it->path().extension() == ".framemap.log" )
       const char *  stringthing = it->path().filename().c_str();

        if ( boost::regex_match( stringthing,what , framelogfilter) )
        {
          ReturnStrings.push_back(it->path().c_str());
        }
     }
   }
  return ReturnStrings;
}
void VideoTrackerMatching::FindTrackingMatrixDirectories()
{
  boost::filesystem::recursive_directory_iterator end_itr;
  boost::regex IGITrackerFilter ( "(QmitkIGITrackerSource)");
  for ( boost::filesystem::recursive_directory_iterator it(m_Directory); 
      it != end_itr ; ++it)
   {
     if ( boost::filesystem::is_directory (it->status()) )
     {
       boost::filesystem::recursive_directory_iterator end_itr1;
       for ( boost::filesystem::recursive_directory_iterator it1(it->path());
                          it1 != end_itr ; ++it1)
       {
         if ( boost::filesystem::is_directory (it1->status()) )
         {
           m_TrackingMatrixDirectories.push_back(it1->path().c_str());
           //need to init tracking matrix vector
           TrackingMatrices TempMatrices; 
           m_TrackingMatrices.push_back(TempMatrices);
         }
       }
     }
   }
  return;
}
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
      //  if ( it->path().extension() == ".framemap.log" )
      const char *  stringthing = it->path().filename().c_str();
      if ( boost::regex_match( stringthing,what , TimeStampFilter) )
      {
        ReturnStamps.m_TimeStamps.push_back(strtoul(it->path().filename().stem().c_str(),NULL,10));
      }
    }
  }
  //sort the vector
  std::sort ( ReturnStamps.m_TimeStamps.begin() , ReturnStamps.m_TimeStamps.end());
  return ReturnStamps;
}

void VideoTrackerMatching::ProcessFrameMapFile (std::string filename)
{
  std::ifstream fin(filename.c_str());
  if ( !fin )
  {
    MITK_WARN << "Failed to open frame map file " << filename;
    return;
  }

  std::string line;
  unsigned int frameNumber; 
  unsigned int SequenceNumber;
  unsigned int channel;
  unsigned long TimeStamp;
  unsigned int linenumber = 0;
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
          long * timingError = new long;
          unsigned long TargetTimeStamp = m_TrackingMatrixTimeStamps[i].GetNearestTimeStamp(TimeStamp,timingError);
          
          m_TrackingMatrices[i].m_TimingErrors.push_back(*timingError);

          std::string MatrixFileName = boost::lexical_cast<std::string>(TargetTimeStamp) + ".txt";
          boost::filesystem::path MatrixFileNameFull (m_TrackingMatrixDirectories[i]);
          MatrixFileNameFull /= MatrixFileName;

          m_TrackingMatrices[i].m_TrackingMatrices.push_back(ReadTrackerMatrix(MatrixFileNameFull.c_str()));

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
  MITK_INFO << "Read " << linenumber << " lines from " << filename;
    
}

unsigned long TrackingMatrixTimeStamps::GetNearestTimeStamp (unsigned long timestamp,long * Delta)
{
  std::vector<unsigned long>::iterator upper = std::upper_bound (m_TimeStamps.begin() , m_TimeStamps.end(), timestamp);
  std::vector<unsigned long>::iterator lower = std::lower_bound (m_TimeStamps.begin() , m_TimeStamps.end(), timestamp);
  long deltaUpper = *upper - timestamp ;
  long deltaLower = timestamp - *lower ;
  long returnValue;
  long delta;
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

cv::Mat VideoTrackerMatching::ReadTrackerMatrix(std::string filename)
{
  cv::Mat TrackerMatrix = cv::Mat(4,4, CV_64FC1);
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
      fin >> TrackerMatrix.at<double>(row,col);
    }
  }
  return TrackerMatrix;
}
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
    double stddev = 0 ;
    long minimum = m_TrackingMatrices[i].m_TimingErrors[0];
    long maximum = m_TrackingMatrices[i].m_TimingErrors[0];

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
} // namespace

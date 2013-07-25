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
        MITK_INFO << "Found tracking directory " << m_TrackingMatrixDirectories[i];
        //find all the files in it and stick em in a vector
        
        TrackingMatrixTimeStamps tempTimeStamps = FindTrackingTimeStamps(m_TrackingMatrixDirectories[i]);
        MITK_INFO << "Found " << tempTimeStamps.m_TimeStamps.size() << " time stamped files";
        m_TrackingMatrixTimeStamps.push_back(tempTimeStamps);
      }
    }
  }

  ProcessFrameMapFile(FrameMaps[0]);

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
          
          TrackingMatrices TempMatrices; 
          TempMatrices.m_TimingErrors.push_back( *timingError );

          std::string MatrixFileName = boost::lexical_cast<std::string>(TargetTimeStamp) + ".txt";
          boost::filesystem::path MatrixFileNameFull (m_TrackingMatrixDirectories[i]);
          MatrixFileNameFull /= MatrixFileName;

          TempMatrices.m_TrackingMatrices.push_back(ReadTrackerMatrix(MatrixFileNameFull.c_str()));

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
} // namespace

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
    std::vector<std::string> TrackingDirectories = FindTrackingMatrixDirectories();
    if ( TrackingDirectories.size() == 0 ) 
    {
      MITK_ERROR << "Found no tracking directories, VideoTrackerMatching failed to initiliase.";
      m_Ready=false;
      return;
    }
    else 
    {
      for ( unsigned int i = 0 ; i < TrackingDirectories.size() ; i ++ ) 
      {
        MITK_INFO << "Found tracking directory " << TrackingDirectories[i];
        //find all the files in it and stick em in a vector
        
        TrackingMatrixTimeStamps tempTimeStamps = FindTrackingTimeStamps(TrackingDirectories[i]);
        MITK_INFO << "Found " << tempTimeStamps.m_TimeStamps.size() << " time stamped files";
        long * delta = new  long();
        MITK_INFO << tempTimeStamps.GetNearestTimeStamp(1374066239633717600, delta);
        MITK_INFO << *delta;
      }
    }

    //now find tracking matrix time stamps


  }

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
std::vector<std::string> VideoTrackerMatching::FindTrackingMatrixDirectories()
{
  boost::filesystem::recursive_directory_iterator end_itr;
  boost::regex IGITrackerFilter ( "(QmitkIGITrackerSource)");
  std::vector<std::string> ReturnStrings;
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
           ReturnStrings.push_back(it1->path().c_str());
         }
       }
     }
   }
  return ReturnStrings;
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
    if ( deltaLower < deltaUpper ) 
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

} // namespace

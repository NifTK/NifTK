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
} // namespace

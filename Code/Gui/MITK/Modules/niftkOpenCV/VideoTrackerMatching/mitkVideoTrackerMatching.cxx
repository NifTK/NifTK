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
{}

VideoTrackerMatching::~VideoTrackerMatching () 
{}

void VideoTrackerMatching::Initialise(std::string directory)
{
  boost::filesystem::recursive_directory_iterator end_itr;
  //boost::regex framelogfilter ( "(*)", boost::regex::basic);
  boost::regex framelogfilter ( "(.+)(framemap.log)");
  for ( boost::filesystem::recursive_directory_iterator it(directory); 
      it != end_itr ; ++it)
   {
     if ( boost::filesystem::is_regular_file (it->status()) )
     {
       boost::cmatch what;
      //  if ( it->path().extension() == ".framemap.log" )
       const char *  stringthing = it->path().filename().c_str();

        if ( boost::regex_match( stringthing,what , framelogfilter) )
        {
          MITK_INFO << "Found " << it->path().filename();
        }
     }
   }


}
} // namespace

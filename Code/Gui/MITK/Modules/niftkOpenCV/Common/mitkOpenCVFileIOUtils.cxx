/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkOpenCVFileIOUtils.h"
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <mitkLogMacros.h>
#include <mitkExceptionMacro.h>

namespace mitk {

//---------------------------------------------------------------------------
bool CheckIfDirectoryContainsTrackingMatrices(const std::string& directory)
{
  boost::regex timeStampFilter ( "([0-9]{19})(.txt)");
  boost::filesystem::directory_iterator endItr;
  
  for ( boost::filesystem::directory_iterator it(directory); it != endItr ; ++it)
  {
    if ( boost::filesystem::is_regular_file (it->status()) )
    {
      boost::cmatch what;
      std::string stringthing = it->path().filename().string();
      if ( boost::regex_match( stringthing.c_str(), what, timeStampFilter) )
      {
        return true;
      }
    }
  }
  return false;
}


//---------------------------------------------------------------------------
std::vector<std::string> FindTrackingMatrixDirectories(const std::string& directory)
{
  std::vector<std::string> directories;
  boost::filesystem::recursive_directory_iterator endItr;
  
  for ( boost::filesystem::recursive_directory_iterator it(directory); it != endItr ; ++it)
  {
    if ( boost::filesystem::is_directory (it->status()) )
    {
      if ( CheckIfDirectoryContainsTrackingMatrices(it->path().string()))
      {
         directories.push_back(it->path().string());
      }
    }
  }
  std::sort (directories.begin(), directories.end());
  return directories;
}


//---------------------------------------------------------------------------
mitk::TrackingMatrixTimeStamps FindTrackingTimeStamps(std::string directory)
{
  boost::filesystem::directory_iterator endItr;
  boost::regex timeStampFilter ( "([0-9]{19})(.txt)");
  TrackingMatrixTimeStamps returnStamps;
  
  for ( boost::filesystem::directory_iterator it(directory);it != endItr ; ++it)
  {
    if ( boost::filesystem::is_regular_file (it->status()) )
    {
      boost::cmatch what;
      std::string stringThing = it->path().filename().string();
      if ( boost::regex_match( stringThing.c_str(), what, timeStampFilter) )
      {
        returnStamps.m_TimeStamps.push_back(boost::lexical_cast<unsigned long long>(it->path().filename().stem().string().c_str()));
      }
    }
  }
  std::sort ( returnStamps.m_TimeStamps.begin() , returnStamps.m_TimeStamps.end());
  return returnStamps;
}


//---------------------------------------------------------------------------
std::vector<std::string> FindVideoFrameMapFiles(const std::string directory)
{
  boost::filesystem::recursive_directory_iterator endItr;
  boost::regex frameLogFilter ( "(.+)(framemap.log)");
  std::vector<std::string> returnStrings;

  for ( boost::filesystem::recursive_directory_iterator it(directory); it != endItr ; ++it)
  {
    if ( boost::filesystem::is_regular_file (it->status()) )
    {
      boost::cmatch what;
      const std::string stringThing = it->path().filename().string();

      if ( boost::regex_match( stringThing.c_str(), what, frameLogFilter) )
      {
        returnStrings.push_back(it->path().string());
      }
    }
  }
  return returnStrings;
}


//---------------------------------------------------------------------------
bool ReadTrackerMatrix(const std::string& filename, cv::Mat& outputMatrix)
{
  bool isSuccessful = false;
  if (outputMatrix.rows != 4)
  {
    mitkThrow() << "ReadTrackerMatrix: Matrix does not have 4 rows" << std::endl;
  }
  if (outputMatrix.cols != 4)
  {
    mitkThrow() << "ReadTrackerMatrix: Matrix does not have 4 columns" << std::endl;
  }

  std::ifstream fin(filename.c_str());
  if ( !fin )
  {
    MITK_WARN << "ReadTrackerMatrix: Failed to open matrix file " << filename;
    return isSuccessful;
  }

  for ( int row = 0 ; row < 4 ; row ++ )
  {
    for ( int col = 0 ; col < 4 ; col ++ ) 
    {
      fin >> outputMatrix.at<double>(row,col);
    }
  }
  isSuccessful = true;
  return isSuccessful;
}

} // end namespace

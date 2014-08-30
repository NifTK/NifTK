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
#include <niftkFileHelper.h>

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


//---------------------------------------------------------------------------
bool SaveTrackerMatrix(const std::string& filename, cv::Mat& outputMatrix)
{
  bool isSuccessful = false;
  if (outputMatrix.rows != 4)
  {
    mitkThrow() << "SaveTrackerMatrix: Matrix does not have 4 rows" << std::endl;
  }
  if (outputMatrix.cols != 4)
  {
    mitkThrow() << "SaveTrackerMatrix: Matrix does not have 4 columns" << std::endl;
  }

  std::ofstream fout(filename.c_str());
  if ( !fout )
  {
    MITK_WARN << "SaveTrackerMatrix: Failed to open matrix file " << filename;
    return isSuccessful;
  }
  for ( int row = 0 ; row < 4 ; row ++ )
  {
    for ( int col = 0 ; col < 4 ; col ++ ) 
    {
      fout << outputMatrix.at<double>(row,col);
      if ( col < 3 ) 
      {
        fout << " ";
      }
    }
    if ( row < 3 ) 
    {
      fout << std::endl;
    }
  }
  fout.close();
  isSuccessful = true;
  return isSuccessful;
}

//---------------------------------------------------------------------------
cv::VideoCapture* InitialiseVideoCapture ( std::string filename , bool ignoreErrors )
{
  cv::VideoCapture* capture = new cv::VideoCapture (filename);
  if ( ! capture )
  {
    mitkThrow() << "Failed to open " << filename << " for video capture" << std::endl;
  }
  //try and get some information about the capture, if these calls fail it may be that 
  //the capture may still work but will exhibit undesirable behaviour, see trac 3718
  int m_VideoWidth = capture->get(CV_CAP_PROP_FRAME_WIDTH);
  int m_VideoHeight = capture->get(CV_CAP_PROP_FRAME_HEIGHT);

  if ( m_VideoWidth == 0 || m_VideoHeight == 0 )
  {
    if ( ! ignoreErrors )
    {
      mitkThrow() << "Problem opening video file for capture. You may want to try rebuilding openCV with ffmpeg support or if available and you're feeling brave over riding video read errors with an ignoreVideoErrors parameter.";
    }
    else 
    {
      MITK_WARN << "mitk::InitialiseVideo detected errors with video file decoding but persevering any way as ignoreErrors is set true";
    }
  }

  return capture;
}


//---------------------------------------------------------------------------
std::vector< std::pair<unsigned long long, cv::Point3d> > LoadTimeStampedPoints(const std::string& directory)
{
  std::vector< std::pair<unsigned long long, cv::Point3d> > timeStampedPoints;

  std::vector<std::string> pointFiles = niftk::GetFilesInDirectory(directory);
  std::sort(pointFiles.begin(), pointFiles.end());

  for (unsigned int i = 0; i < pointFiles.size(); i++)
  {
    cv::Point3d point;
    std::string fileName = pointFiles[i];

    if(fileName.size() > 0)
    {
      std::ifstream myfile(fileName.c_str());
      if (myfile.is_open())
      {
        point.x = 0;
        point.y = 0;
        point.z = 0;

        myfile >> point.x;
        myfile >> point.y;
        myfile >> point.z;

        if (myfile.bad() || myfile.eof() || myfile.fail())
        {
          std::ostringstream errorMessage;
          errorMessage << "Could not load point file:" << fileName << std::endl;
          mitkThrow() << errorMessage.str();
        }
        myfile.close();
      }
    }

    // Parse timestamp.
    boost::regex timeStampFilter ( "([0-9]{19})(.txt)");
    boost::cmatch what;
    unsigned long long timeStamp = 0;

    if ( boost::regex_match( (niftk::Basename(fileName)).c_str(), what, timeStampFilter) )
    {
      timeStamp = boost::lexical_cast<unsigned long long>(niftk::Basename(fileName));

      if (timeStamp != 0)
      {
        timeStampedPoints.push_back(std::pair<unsigned long long, cv::Point3d>(timeStamp, point));
      }
      else
      {
        std::ostringstream errorMessage;
        errorMessage << "Failed to extract timestamp from name of file:" << fileName << std::endl;
        mitkThrow() << errorMessage.str();
      }
    }
    else
    {
      std::ostringstream errorMessage;
      errorMessage << "Could not match timestamp in name of file:" << fileName << std::endl;
      mitkThrow() << errorMessage.str();
    }
  }

  return timeStampedPoints;
}

} // end namespace

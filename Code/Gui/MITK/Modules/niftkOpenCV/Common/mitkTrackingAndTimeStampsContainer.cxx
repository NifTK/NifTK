/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTrackingAndTimeStampsContainer.h"
#include <algorithm>
#include <niftkFileHelper.h>
#include <mitkOpenCVFileIOUtils.h>
#include <mitkOpenCVMaths.h>
#include <mitkExceptionMacro.h>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <sstream>
#include <vector>

namespace mitk {

//-----------------------------------------------------------------------------
void TrackingAndTimeStampsContainer::Clear()
{
  m_TimeStamps.Clear();
  m_TrackingMatrices.clear();
}


//-----------------------------------------------------------------------------
int TrackingAndTimeStampsContainer::LoadFromDirectory(const std::string& dirName)
{
  int numberLoaded = 0;

  if (!CheckIfDirectoryContainsTrackingMatrices(dirName))
  {
    std::ostringstream errorMessage;
    errorMessage << "LoadFromDirectory(" << dirName << ") did not find any tracking matrices." << std::endl;
    mitkThrow() << errorMessage.str();
  }

  std::vector<std::string> fileNames;
  boost::filesystem::directory_iterator endItr;
  boost::regex timeStampFilter ( "([0-9]{19})(.txt)");
  boost::regex timeStampFilterNoExtension ( "([0-9]{19})");
  boost::cmatch what;

  // Extract filenames.
  for ( boost::filesystem::directory_iterator it(dirName); it != endItr ; ++it)
  {
    if ( boost::filesystem::is_regular_file (it->status()) )
    {
      std::string stringThing = it->path().filename().string();
      if ( boost::regex_match( stringThing.c_str(), what, timeStampFilter) )
      {
        fileNames.push_back(it->path().string());
      }
    }
  }

  // Sort the filenames.
  std::sort(fileNames.begin(), fileNames.end());

  // Extract time-stamp AND corresponding matrix.
  TimeStampsContainer::TimeStamp timeStamp = 0;
  cv::Matx44d matrix;

  for (unsigned int i = 0; i < fileNames.size(); i++)
  {
    if ( boost::regex_match(niftk::Basename(fileNames[i]).c_str(), what, timeStampFilterNoExtension))
    {
      timeStamp = boost::lexical_cast<TimeStampsContainer::TimeStamp>(niftk::Basename(fileNames[i]));
      if (timeStamp != 0)
      {
        if (mitk::ReadTrackerMatrix(fileNames[i], matrix))
        {
          m_TimeStamps.Insert(timeStamp);
          m_TrackingMatrices.push_back(matrix);
        }
        else
        {
          std::ostringstream errorMessage;
          errorMessage << "Failed to load matrix from file:" << fileNames[i] << std::endl;
          mitkThrow() << errorMessage.str();
        }
      }
      else
      {
        std::ostringstream errorMessage;
        errorMessage << "Failed to extract timestamp from name of file:" << niftk::Basename(fileNames[i]) << std::endl;
        mitkThrow() << errorMessage.str();
      }
    }
    else
    {
      std::ostringstream errorMessage;
      errorMessage << "Could not match timestamp in name of file:" << niftk::Basename(fileNames[i]) << std::endl;
      mitkThrow() << errorMessage.str();
    }
  }
  return numberLoaded;
}

//-----------------------------------------------------------------------------
int TrackingAndTimeStampsContainer::SaveToDirectory(const std::string& dirName)
{
  int numberSaved = 0;
  
  boost::filesystem::path savePath (dirName);
  if ( ! (( boost::filesystem::exists (savePath) ) || ( boost::filesystem::create_directories (savePath)) ) )
  {
    MITK_WARN << "TrackingAndTimeStampsContainer::SaveToDirectory failed to find or make save directory";
    return numberSaved;
  }

  for ( unsigned int i = 0 ; i < m_TrackingMatrices.size() ; i ++ )
  {

    TimeStampsContainer::TimeStamp timeStamp = m_TimeStamps.GetTimeStamp(i);
    cv::Matx44d matrix = m_TrackingMatrices[i];

    std::string fileName = boost::lexical_cast<std::string>(timeStamp);
    std::string fullFileName = dirName + fileName + ".txt";
    mitk::SaveTrackerMatrix(fullFileName, matrix);
    numberSaved++;
  }
  return numberSaved;
}



//-----------------------------------------------------------------------------
void TrackingAndTimeStampsContainer::Insert(const TimeStampsContainer::TimeStamp& timeStamp, const cv::Matx44d& matrix)
{
  m_TimeStamps.Insert(timeStamp);
  m_TrackingMatrices.push_back(matrix);
}


//-----------------------------------------------------------------------------
TimeStampsContainer::TimeStamp TrackingAndTimeStampsContainer::GetTimeStamp(std::vector<TimeStampsContainer::TimeStamp>::size_type frameNumber) const
{
  return m_TimeStamps.GetTimeStamp(frameNumber);
}


//-----------------------------------------------------------------------------
cv::Matx44d TrackingAndTimeStampsContainer::GetMatrix(std::vector<TimeStampsContainer::TimeStamp>::size_type frameNumber) const
{
  assert(frameNumber >= 0);
  assert(frameNumber < m_TrackingMatrices.size());
  return m_TrackingMatrices[frameNumber];
}


//-----------------------------------------------------------------------------
std::vector<TimeStampsContainer::TimeStamp>::size_type TrackingAndTimeStampsContainer::GetSize() const
{
  assert(m_TrackingMatrices.size() == m_TimeStamps.GetSize());
  return m_TimeStamps.GetSize();
}


//-----------------------------------------------------------------------------
std::vector<TimeStampsContainer::TimeStamp>::size_type TrackingAndTimeStampsContainer::GetFrameNumber(const TimeStampsContainer::TimeStamp& timeStamp) const
{
  return m_TimeStamps.GetFrameNumber(timeStamp);
}


//-----------------------------------------------------------------------------
TimeStampsContainer::TimeStamp TrackingAndTimeStampsContainer::GetNearestTimeStamp(const TimeStampsContainer::TimeStamp& timeStamp, long long *delta) const
{
  return m_TimeStamps.GetNearestTimeStamp(timeStamp, delta);
}


//-----------------------------------------------------------------------------
cv::Matx44d TrackingAndTimeStampsContainer::InterpolateMatrix(const TimeStampsContainer::TimeStamp& timeStamp, TimeStampsContainer::TimeStamp& minError, bool& inBounds)
{
  TimeStampsContainer::TimeStamp before;
  TimeStampsContainer::TimeStamp after;
  double proportion = 0;
  inBounds=false;
    
  cv::Matx44d interpolatedMatrix;
  std::vector<TimeStampsContainer::TimeStamp>::size_type indexBefore;
  std::vector<TimeStampsContainer::TimeStamp>::size_type indexAfter;

  if ( m_TrackingMatrices.size() == 0 )
  {
    mitkThrow() << "TrackingAndTimeStampsContainer::InterpolateMatrix There are no tracking matrices set";
  }

  if (m_TimeStamps.GetBoundingTimeStamps(timeStamp, before, after, proportion))
  {
    indexBefore = this->GetFrameNumber(before);
    indexAfter = this->GetFrameNumber(after);

    mitk::InterpolateTransformationMatrix(m_TrackingMatrices[indexBefore], m_TrackingMatrices[indexAfter], proportion, interpolatedMatrix);
    if ( proportion > 0.5 )
    {
      minError = after - timeStamp;
    }
    else
    {
      minError = timeStamp - before;
    }
    inBounds = true;
    return interpolatedMatrix;
  }
  else
  {
    inBounds=false;
    if ( before == 0 ) 
    {
      minError = after - timeStamp;
      indexAfter = this->GetFrameNumber(after);
      return m_TrackingMatrices[indexAfter];
    }
    else
    {
      minError = timeStamp - before;
      indexBefore = this->GetFrameNumber(before);
      return m_TrackingMatrices[indexBefore];
    }
  }
}

} // end namespace

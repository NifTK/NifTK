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
std::vector<TimeStampsContainer::TimeStamp>::size_type TrackingAndTimeStampsContainer::GetFrameNumber(const TimeStampsContainer::TimeStamp& timeStamp)
{
  return m_TimeStamps.GetFrameNumber(timeStamp);
}


//-----------------------------------------------------------------------------
cv::Matx44d TrackingAndTimeStampsContainer::InterpolateMatrix(const TimeStampsContainer::TimeStamp& timeStamp, TimeStampsContainer::TimeStamp& minError)
{
  TimeStampsContainer::TimeStamp before;
  TimeStampsContainer::TimeStamp after;
  double proportion = 0;

  if (m_TimeStamps.GetBoundingTimeStamps(timeStamp, before, after, proportion))
  {
    std::vector<TimeStampsContainer::TimeStamp>::size_type indexBefore;
    std::vector<TimeStampsContainer::TimeStamp>::size_type indexAfter;
    indexBefore = this->GetFrameNumber(before);
    indexAfter = this->GetFrameNumber(after);

    cv::Matx44d interpolatedMatrix;
    mitk::InterpolateTransformationMatrix(m_TrackingMatrices[indexBefore], m_TrackingMatrices[indexAfter], proportion, interpolatedMatrix);
    if ( proportion > 0.5 )
    {
      minError = after - timeStamp;
    }
    else
    {
      minError = timeStamp - before;
    }

    return interpolatedMatrix;
  }
  else
  {
   // that failed so now what?
  }
}

} // end namespace

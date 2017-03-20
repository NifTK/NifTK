/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkTimingCalibrationInterface.h"
#include <niftkNiftyCalTypes.h>
#include <niftkTimingCalibration.h>
#include <cv.h>
#include <vector>
#include <mitkOpenCVFileIOUtils.h>

namespace niftk
{

//-----------------------------------------------------------------------------
std::vector<TimingSample3D> ConvertOpenCVDataToTimingSamples(
  const std::vector< std::pair<unsigned long long, cv::Point3d> >& input)
{
  std::vector<TimingSample3D> output;
  for (size_t i = 0; i < input.size(); i++)
  {
    TimingSample3D op;
    op.time = input[i].first;
    op.sample = input[i].second;

    output.push_back(op);
  }
  return output;
}


//-----------------------------------------------------------------------------
int DoTimingCalibrationFromDirectories(const std::string& trackingDir,
                                       const std::string& pointsDir)
{
  std::vector< std::pair<unsigned long long, cv::Point3d> > trackingData = mitk::LoadTimeStampedTranslations(trackingDir);
  if (trackingData.empty())
  {
    mitkThrow() << "Failed to load tracking data from " << trackingDir << std::endl;
  }

  std::vector< std::pair<unsigned long long, cv::Point3d> > pointData = mitk::LoadTimeStampedPoints(pointsDir);
  if (pointData.empty())
  {
    mitkThrow() << "Failed to load point data from " << pointsDir << std::endl;
  }

  std::vector<TimingSample3D> tData = ConvertOpenCVDataToTimingSamples(trackingData);
  std::vector<TimingSample3D> pData = ConvertOpenCVDataToTimingSamples(pointData);

  int lag = niftk::TimingCalibration(tData, pData);
  return lag;
}

} // end namespace

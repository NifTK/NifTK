/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkTimingCalibrationInterface.h"
#include <cv.h>
#include <vector>
#include <mitkOpenCVFileIOUtils.h>

namespace niftk
{

//-----------------------------------------------------------------------------
int DoTimingCalibrationFromDirectories(const std::string& trackingDir,
                                       const std::string& pointsDir)
{
  int lag = 0;

  std::vector< std::pair<unsigned long long, cv::Point3d> > pointData = mitk::LoadTimeStampedPoints(pointsDir);
  if (pointData.size() == 0)
  {
    mitkThrow() << "Failed to load point data from " << pointsDir << std::endl;
  }

  std::vector< std::pair<unsigned long long, cv::Point3d> > trackingData = mitk::LoadTimeStampedTranslations(trackingDir);
  if (trackingData.size() == 0)
  {
    mitkThrow() << "Failed to load tracking data from " << trackingDir << std::endl;
  }

  return lag;
}

} // end namespace

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkOpenCVFileIOUtils_h
#define mitkOpenCVFileIOUtils_h

#include "niftkOpenCVExports.h"
#include <mitkTrackingMatrixTimeStamps.h>

/**
 * \file mitkOpenCVFileIOUtils.h
 * \brief Various simple file IO utils, that may use open CV data types in their signatures.
 */
namespace mitk {

/**
 * \brief Iterates through a directory to see if it contains any files that have a timestamp as a name, and end in .txt
 */
extern "C++" NIFTKOPENCV_EXPORT bool CheckIfDirectoryContainsTrackingMatrices(const std::string& directory);

/**
 * \brief Recursively hunts for all directories that look like they contain tracking matrices, <timestamp>.txt.
 */
extern "C++" NIFTKOPENCV_EXPORT std::vector<std::string> FindTrackingMatrixDirectories(const std::string& directory);

/**
 * \brief Returns an mitk::TrackingMatrixTimeStamps containing all the timestamps of tracking matrices.
 */
extern "C++" NIFTKOPENCV_EXPORT mitk::TrackingMatrixTimeStamps FindTrackingTimeStamps(std::string directory);

/**
 * \brief Recursively hunts for all files that look like they are a video frame map file, (.+)(framemap.log).
 */
extern "C++" NIFTKOPENCV_EXPORT std::vector<std::string> FindVideoFrameMapFiles(const std::string directory);

/**
 * \brief Extracted from mitkVideoTrackerMatching, reads a 4x4 matrix into a cv::Mat, and if the matrix can't
 * be read, will return a new matrix thats initialised according to the default OpenCV macros (i.e. unitinitialised).
 * @return true if successful and false otherwise
 */
bool ReadTrackerMatrix(const std::string& filename, cv::Mat& outputMatrix);

} // end namespace

#endif




/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkOpenCVImageProcessing_h
#define mitkOpenCVImageProcessing_h

#include "niftkOpenCVUtilsExports.h"
#include <cv.h>

/**
 * \file mitkOpenCVImageProcessing.h
 * \brief Various simplish image processing functions
 */
namespace mitk {

/**
 * \brief finds the intersection of two lines in an image
 */

extern "C++" NIFTKOPENCVUTILS_EXPORT cv::Point2d FindCrosshairCentre(const cv::Mat& image,
  const int& cannyLowThreshold, const int& cannyHighThreshold, const int& cannyKernel, 
  const double& houghRho, const double& houghTheta, const int& houghThreshold, 
  const int& houghLineLength, const int& houghLineGap, cv::vector <cv::Vec4i>& lines);

/** 
 * \brief Applies binary mask to a vector of point pairs
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT unsigned int ApplyMask (
    std::vector < std::pair <cv::Point2d, cv::Point2d> >& pointPairs,
    const cv::Mat& maskImage, const unsigned int& maskValue, const bool& maskUsingFirst);

} // end namespace

namespace niftk
{
/**
 * \brief Takes an image of the liver phantom, assumed to be against a plain black
 * or white or both background, and produces a very basic segmentation.
 */
extern "C++" NIFTKOPENCVUTILS_EXPORT void SegmentLiverPhantom(const std::string& inputFileName,
                                                              const std::string& outputFileName
                                                             );
} // end niftk namespace

#endif




/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkOpenCVImageProcessing.h"
#include <numeric>
#include <algorithm>
#include <functional>
#include <mitkMathsUtils.h>
#include <mitkExceptionMacro.h>
#include <mitkOpenCVMaths.h>
#include <opencv2/gpu/gpu.hpp>

namespace mitk {

//-----------------------------------------------------------------------------
cv::Point2d FindCrosshairCentre(const cv::Mat& image, 
    const int& cannyLowThreshold, const int& cannyHighThreshold, const int& cannyKernel,
    const double& houghRho, const double& houghTheta, const int& houghThreshold, 
    const int& houghLineLength, const int& houghLineGap , cv::vector <cv::Vec4i>& lines)
{
  assert ( image.type() == CV_8UC3 || image.type() == CV_8UC4 || image.type() == CV_8UC1 );
  // is open cv built with CUDA support ?
  if ( cv::gpu::getCudaEnabledDeviceCount () != 0 ) 
  {
    MITK_INFO << "Found " <<  cv::gpu::getCudaEnabledDeviceCount () << " CUDA devices." ;
    //we could switch to a GPU implementation, which might be faster
  }

  cv::Mat working1;
  cv::Mat working2;
  if ( image.type() == CV_8UC4 )
  {
    cv::cvtColor (image , working1, CV_BGRA2GRAY );
  }
  if ( image.type() == CV_8UC3 )
  {  
    cv::cvtColor ( image, working1 , CV_BGR2GRAY );
  }
  if ( image.type() == CV_8UC1 )
  {
    working1 = image;
  }
  assert ( working1.type() == CV_8UC1 );

  cv::Canny ( working1, working2, cannyLowThreshold, cannyHighThreshold, cannyKernel);
  cv::HoughLinesP ( working2, lines, houghRho, houghTheta, houghThreshold, houghLineLength, houghLineGap);
  std::vector <cv::Point2d> intersections = mitk::FindIntersects (lines, true, true, 60);

  working1.release();
  working2.release();
  return mitk::GetCentroid ( intersections, true);
}

} // end namespace

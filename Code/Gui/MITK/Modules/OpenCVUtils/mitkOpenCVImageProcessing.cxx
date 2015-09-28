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
#include <niftkMathsUtils.h>
#include <mitkExceptionMacro.h>
#include <mitkOpenCVMaths.h>
#include <opencv2/gpu/gpu.hpp>
#include <boost/math/special_functions/round.hpp>

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

//-----------------------------------------------------------------------------
class in_mask
{
  const cv::Mat m_Mask;
  const unsigned char m_BlankValue;
  const bool m_UseFirstValue;

public:
  in_mask ( const cv::Mat& mask, const unsigned int& blankValue, const bool& useFirstValue)
  : m_Mask (mask)
  , m_BlankValue (blankValue )
  , m_UseFirstValue (useFirstValue)
  {
    assert ( mask.depth() == CV_8U );
    assert ( mask.channels() == 1 );
  }

  bool operator () ( const std::pair<cv::Point2d, cv::Point2d> & pointPair ) const
  {
    unsigned char maskValue = m_BlankValue;
    cv::Point2i point;
    if ( m_UseFirstValue )
    {
      point.x = boost::math::round (pointPair.first.x );
      point.y = boost::math::round (pointPair.first.y );
    }
    else
    {
      point.x = boost::math::round (pointPair.second.x );
      point.y = boost::math::round (pointPair.second.y );
    }

    if ( (point.x >= 0) &&
        (point.x < m_Mask.cols) &&
        (point.y >= 0) &&
        (point.y < m_Mask.rows) )
    {
      maskValue = m_Mask.at<unsigned char> ( point.y, point.x );
    }

    if ( maskValue == m_BlankValue )
    {
      return true;
    }
    else
    {
      return false;
    }
  }
};

//-----------------------------------------------------------------------------
unsigned int ApplyMask ( std::vector <std::pair < cv::Point2d, cv::Point2d > >& pointPairs, const cv::Mat& mask, 
    const unsigned int& blankValue, const bool& maskUsingFirst )
{
  unsigned int originalSize = pointPairs.size();
  pointPairs.erase ( std::remove_if ( pointPairs.begin(), pointPairs.end(), in_mask(mask, blankValue, maskUsingFirst)  ), pointPairs.end());
  return originalSize - pointPairs.size();
}

} // end namespace

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkSplitVideo.h"
#include <cv.h>
#include <highgui.h>

namespace mitk {

//-----------------------------------------------------------------------------
SplitVideo::SplitVideo()
{

}


//-----------------------------------------------------------------------------
SplitVideo::~SplitVideo()
{

}


//-----------------------------------------------------------------------------
bool SplitVideo::Split(
    const std::string& inputImageBaseName,
    const unsigned int& startFrame,
    const unsigned int& endFrame)
{
  bool isSuccessful = false;

  std::vector <std::string> videoFiles = niftk::FindVideoData(inputImageBaseName);
  CvCapture*      capturer = cvCreateFileCapture(inputImageBaseName.c_str());
  //FIX ME, this should not be hard coded
  cv::Size S = cv::Size((int) 960, (int) 270 );
  CvVideoWriter*  writerer = cvCreateVideoWriter("out.264",CV_FOURCC('D','I','V','X'),60,S, true);

  

  return isSuccessful;
}

} // end namespace

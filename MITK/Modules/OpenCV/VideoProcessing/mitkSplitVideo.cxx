/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkSplitVideo.h"
#include <niftkFileHelper.h>
#include <mitkOpenCVFileIOUtils.h>
#include <cv.h>
#include <highgui.h>
#include <fstream>

#include <boost/lexical_cast.hpp>

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
  std::vector <std::string> frameMapFiles = mitk::FindVideoFrameMapFiles(inputImageBaseName);
  for ( unsigned int i = 0 ; i < videoFiles.size() ; i ++ ) 
  {
    MITK_INFO << "found " << videoFiles[i];
  }
  for ( unsigned int i = 0 ; i < frameMapFiles.size() ; i ++ ) 
  {
    MITK_INFO << "found " << frameMapFiles[i];
  }
  if ( videoFiles.size() != 1 || frameMapFiles.size() != 1 ) 
  {
    MITK_ERROR << "found the wrong number of video or frame map files, quitting";
    return false;
  }

  cv::VideoCapture*  capturer;
  std::string videoIn = videoFiles[0];
  MITK_INFO << "trying to open " << videoIn;
  capturer = mitk::InitialiseVideoCapture(videoIn);

  std::ifstream fin(frameMapFiles[0].c_str());
  if ( !fin )
  {
    MITK_WARN << "Failed to open frame map file " << frameMapFiles[0];
    return false;
  }

  cv::Size S = cv::Size (static_cast<int> (capturer->get(CV_CAP_PROP_FRAME_WIDTH)),
    static_cast<int>(capturer->get(CV_CAP_PROP_FRAME_HEIGHT)) );
  double fps = static_cast<double>(capturer->get(CV_CAP_PROP_FPS));
  int codec = static_cast<int>(capturer->get(CV_CAP_PROP_FOURCC));

  long EXT[] = {codec & 0XFF, (codec & 0XFF00) >> 8, (codec & 0XFF0000) >> 16, (codec & 0XFF000000) >> 24, 0};
  MITK_INFO << codec << " " << EXT[0] << ", " << EXT[1] << ", " << EXT[2] << ", " << EXT[3];

  std::string outVideoName = videoIn + "." +  boost::lexical_cast<std::string>(startFrame)
    + "-" +  boost::lexical_cast<std::string>(endFrame) + ".avi";
  std::string outLogName = videoIn + "." +  boost::lexical_cast<std::string>(startFrame)
    + "-" +  boost::lexical_cast<std::string>(endFrame) + ".framemap.log";

  cv::VideoWriter* writerer = mitk::CreateVideoWriter(outVideoName.c_str(),fps,S);

  std::ofstream fout(outLogName.c_str());

  std::string line;
  unsigned int frameNumber = 0;
  unsigned int sequenceNumber;
  unsigned int channel;
  unsigned long long timeStamp;
  unsigned int videoFrameNumber = 0;
  unsigned int videoOutFrameNumber = 0;

  while ( std::getline(fin,line) && ( frameNumber <= endFrame ) )
  {
    if ( line[0] != '#' )
    {
      std::stringstream linestream(line);
      linestream >> frameNumber >> sequenceNumber >> channel >> timeStamp;
      if ( !linestream.fail() ) 
      {
        cv::Mat videoImage;
        capturer->read(videoImage);

        if ( frameNumber != videoFrameNumber ) 
        {
          MITK_ERROR << "framemap frame number not equal to video frame number, halting. [ " <<
            frameNumber << " != " << videoFrameNumber;
          exit(1);
        }

        if ( frameNumber >= startFrame )
        {
          MITK_INFO << "Writing input frame " << frameNumber << " to output Frame " << videoOutFrameNumber;
          writerer->write(videoImage);
          fout << videoOutFrameNumber << "\t" << sequenceNumber << "\t" << channel << "\t" << timeStamp << std::endl;
          videoOutFrameNumber ++;
        }
        videoFrameNumber ++;
      }
      else 
      {
        MITK_ERROR << "Parse Failure";
      }
    }
  }

 fout.close();
 fin.close();
 capturer->release();
 writerer->release();

  return isSuccessful;
}

} // end namespace

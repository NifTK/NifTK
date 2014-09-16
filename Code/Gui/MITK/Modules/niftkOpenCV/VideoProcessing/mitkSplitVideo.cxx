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

  CvCapture*  capturer;
  std::string videoIn = videoFiles[0];
  MITK_INFO << "trying to open " << videoIn;
  capturer = cvCreateFileCapture(videoIn.c_str());

  std::ifstream fin(frameMapFiles[0].c_str());
  if ( !fin )
  {
    MITK_WARN << "Failed to open frame map file " << frameMapFiles[0];
    return false;
  }

  cv::Size S = cv::Size((int)cvGetCaptureProperty (capturer, CV_CAP_PROP_FRAME_WIDTH),
    (int)cvGetCaptureProperty (capturer, CV_CAP_PROP_FRAME_HEIGHT)) ;
  double fps = (double)cvGetCaptureProperty (capturer, CV_CAP_PROP_FPS);
  int codec = (int)cvGetCaptureProperty (capturer,CV_CAP_PROP_FOURCC);

  char EXT[] = {codec & 0XFF , (codec & 0XFF00) >> 8,(codec & 0XFF0000) >> 16,(codec & 0XFF000000) >> 24, 0};
  MITK_INFO << codec << " " << EXT[0] << ", " << EXT[1] << ", " << EXT[2] << ", " << EXT[3];


  std::string outVideoName = videoIn + "." +  boost::lexical_cast<std::string>(startFrame)
    + "-" +  boost::lexical_cast<std::string>(endFrame) + ".avi";
  std::string outLogName = videoIn + "." +  boost::lexical_cast<std::string>(startFrame)
    + "-" +  boost::lexical_cast<std::string>(endFrame) + ".framemap.log";
  //CvVideoWriter*  writerer = cvCreateVideoWriter(outVideoName.c_str(),codec,fps,S, true);
  CvVideoWriter*  writerer = cvCreateVideoWriter(outVideoName.c_str(),CV_FOURCC('D','I','V','X'),fps,S, true);
  //CV_FOURCC('D', 'I', 'V', 'X')
  //CV_FOURCC('U', '2', '6', '3')
  std::ofstream fout(outLogName.c_str());

  std::string line;
  unsigned int frameNumber = 0;
  unsigned int sequenceNumber;
  unsigned int channel;
  unsigned long long timeStamp;
  unsigned int videoFrameNumber = 0;
  unsigned int videoOutFrameNumber = 0;

  while ( getline(fin,line) && ( frameNumber <= endFrame ) )
  {
    if ( line[0] != '#' )
    {
      std::stringstream linestream(line);
      bool parseSuccess = linestream >> frameNumber >> sequenceNumber >> channel >> timeStamp;
      if ( parseSuccess ) 
      {
        cv::Mat videoImage = cvQueryFrame ( capturer );

        if ( frameNumber != videoFrameNumber ) 
        {
          MITK_ERROR << "framemap frame number not equal to video frame number, halting. [ " <<
            frameNumber << " != " << videoFrameNumber;
          exit(1);
        }

        if ( frameNumber >= startFrame )
        {
          MITK_INFO << "Writing input frame " << frameNumber << " to output Frame " << videoOutFrameNumber;
          IplImage image(videoImage);
          cvWriteFrame(writerer,&image);
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
 cvReleaseVideoWriter(&writerer);

  return isSuccessful;
}

} // end namespace

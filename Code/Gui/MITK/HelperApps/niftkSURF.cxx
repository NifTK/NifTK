/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <niftkSiftCLP.h>
#include <mitkExceptionMacro.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2/nonfree/features2d.hpp>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if (    input.length() == 0
       || output.length() == 0
       )
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    cv::Mat inputImage = cv::imread(input);
    cv::Mat outputImage;
    cv::Mat inputResize;

    /*
    cv::Mat greyImage;
    cv::cvtColor( inputImage, greyImage, CV_BGRA2GRAY );
    cv::resize(greyImage, inputResize, cv::Size(1920, 1080), 0,0, cv::INTER_LANCZOS4);
    cv::cvtColor( inputResize, outputImage, CV_GRAY2BGR );
    std::vector<cv::Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 5;
    bool useHarrisDetector = false;
    double k = 0.04;
    int maxCorners = 5000;

    /// Apply corner detection
    goodFeaturesToTrack(inputResize, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k);

    /// Draw corners detected
    std::cout<<"** Number of corners detected: " << corners.size() << std::endl;
    int r = 10;
    for( int i = 0; i < corners.size(); i++ )
    {
      cv::circle( outputImage, corners[i], r, cv::Scalar(0, 0, 255), 1, 8, 0 );
    }
    */

    cv::SURF detector;
    std::vector<cv::KeyPoint> keypoints;
    cv::resize(inputImage, inputResize, cv::Size(1920, 1080), 0,0, cv::INTER_LANCZOS4);
    detector.detect(inputResize, keypoints);
    cv::drawKeypoints(inputResize, keypoints, outputImage);

    cv::imwrite(output, outputImage);
    returnStatus = EXIT_SUCCESS;
  }
  catch (mitk::Exception& e)
  {
    MITK_ERROR << "Caught mitk::Exception: " << e.GetDescription() << ", from:" << e.GetFile() << "::" << e.GetLine() << std::endl;
    returnStatus = EXIT_FAILURE + 1;
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception: " << e.what() << std::endl;
    returnStatus = EXIT_FAILURE + 2;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:" << std::endl;
    returnStatus = EXIT_FAILURE + 3;
  }

  return returnStatus;
}

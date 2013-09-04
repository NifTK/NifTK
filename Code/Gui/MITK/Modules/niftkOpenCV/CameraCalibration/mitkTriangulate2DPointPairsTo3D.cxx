/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTriangulate2DPointPairsTo3D.h"
#include "mitkCameraCalibrationFacade.h"
#include <ios>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cv.h>
#include <highgui.h>
#include <niftkFileHelper.h>

namespace mitk {

//-----------------------------------------------------------------------------
Triangulate2DPointPairsTo3D::Triangulate2DPointPairsTo3D()
{

}


//-----------------------------------------------------------------------------
Triangulate2DPointPairsTo3D::~Triangulate2DPointPairsTo3D()
{

}



//-----------------------------------------------------------------------------
bool Triangulate2DPointPairsTo3D::Triangulate(const std::string& input2DPointPairsFileName,
                                              const std::string& intrinsicLeftFileName,
                                              const std::string& intrinsicRightFileName,
                                              const std::string& rightToLeftExtrinsics
                                             )
{
  bool isSuccessful = false;

  try
  {
    // Load the pairs of 2D points.
    std::ifstream reader(input2DPointPairsFileName.c_str());
    if (!reader)
    {
      std::cerr << "Failed to open " << input2DPointPairsFileName << std::endl;
      return false;
    }
    else
    {
      std::cout << "Opened " << input2DPointPairsFileName << std::endl;
    }

    std::vector< std::pair<cv::Point2f, cv::Point2f> > pointPairs;
    double numbersOnLine[4];

    while(!reader.eof())
    {
      int i = 0;
      for (i = 0; i < 4; i++)
      {
        reader >> numbersOnLine[i];
      }
      if (reader.fail() || i != 4)
      {
        break;
      }
      cv::Point2f leftPoint;
      cv::Point2f rightPoint;

      leftPoint.x = numbersOnLine[0];
      leftPoint.y = numbersOnLine[1];
      rightPoint.x = numbersOnLine[2];
      rightPoint.y = numbersOnLine[3];

      pointPairs.push_back(std::pair<cv::Point2f, cv::Point2f>(leftPoint, rightPoint));
    }
    reader.close();

    if (pointPairs.size() == 0)
    {
      throw std::logic_error("Failed to read 3D points");
    }

    std::cout << "Triangulate2DPointPairsTo3D: Read in " << pointPairs.size() << " point pairs." << std::endl;

    cv::Mat leftIntrinsic = cvCreateMat (3,3,CV_64FC1);
    cv::Mat leftDistortion = cvCreateMat (1,5,CV_64FC1);
    cv::Mat rightIntrinsic = cvCreateMat (3,3,CV_64FC1);
    cv::Mat rightDistortion = cvCreateMat (1,5,CV_64FC1);
    cv::Mat rightToLeftRotationMatrix = cvCreateMat (3,3,CV_64FC1);
    cv::Mat rightToLeftTranslationVector = cvCreateMat (1,3,CV_64FC1);

    // Load matrices. These throw exceptions if things fail.
    LoadCameraIntrinsicsFromPlainText(intrinsicLeftFileName, &leftIntrinsic, &leftDistortion);
    LoadCameraIntrinsicsFromPlainText(intrinsicRightFileName, &rightIntrinsic, &rightDistortion);
    LoadStereoTransformsFromPlainText(rightToLeftExtrinsics, &rightToLeftRotationMatrix, &rightToLeftTranslationVector);

    // Triangulate each point.
    std::vector< cv::Point3f > pointsIn3D;
    for (unsigned int i = 0; i < pointPairs.size(); i++)
    {
      cv::Point3f pointIn3D = mitk::TriangulatePointPair(
          pointPairs[i],
          leftIntrinsic,
          rightIntrinsic,
          rightToLeftRotationMatrix,
          rightToLeftTranslationVector
          );
      pointsIn3D.push_back(pointIn3D);
    }

    // Print to output for now.
    for (unsigned int i = 0; i < pointsIn3D.size(); i++)
    {
      std::cout << "[" << i << "], 2Dl=(" << pointPairs[i].first.x << ", " << pointPairs[i].first.y << "), 2Dr=(" << pointPairs[i].second.x << ", " << pointPairs[i].second.y << "), 3D=" << pointsIn3D[i].x << ", " << pointsIn3D[i].y << ", " << pointsIn3D[i].z <<  std::endl;
    }

    isSuccessful = true;

  }
  catch(std::logic_error& e)
  {
    std::cerr << "Triangulate2DPointPairsTo3D::Project: exception thrown e=" << e.what() << std::endl;
  }

  return isSuccessful;
}

} // end namespace

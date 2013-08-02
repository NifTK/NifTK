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
#include <FileHelper.h>

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
                                              const std::string& rightToLeftRotationFileName,
                                              const std::string& rightToLeftTranslationFileName
                                             )
{
  bool isSuccessful = false;

  try
  {
    CvMat *intrinsicLeft = (CvMat*)cvLoad(intrinsicLeftFileName.c_str());
    if (intrinsicLeft == NULL)
    {
      throw std::logic_error("Failed to load left camera intrinsic params");
    }
    CvMat *intrinsicRight = (CvMat*)cvLoad(intrinsicRightFileName.c_str());
    if (intrinsicRight == NULL)
    {
      throw std::logic_error("Failed to load right camera intrinsic params");
    }
    CvMat *rightToLeftRotation = (CvMat*)cvLoad(rightToLeftRotationFileName.c_str());
    if (rightToLeftRotation == NULL)
    {
      throw std::logic_error("Failed to load right to left rotation params");
    }

    CvMat *rightToLeftTranslation = (CvMat*)cvLoad(rightToLeftTranslationFileName.c_str());
    if (rightToLeftTranslation == NULL)
    {
      throw std::logic_error("Failed to load right to left translation params");
    }

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
      cv::Point3f pointIn3D;

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

    cv::Mat leftCameraIntrinsicParams(intrinsicLeft);
    cv::Mat rightCameraIntrinsicParams(intrinsicRight);
    cv::Mat rightToLeftRotationVector(rightToLeftRotation);
    cv::Mat rightToLeftTranslationVector(rightToLeftTranslation);

    std::vector< cv::Point3f > pointsIn3D = mitk::TriangulatePointPairsUsingGeometry(
        pointPairs,
        leftCameraIntrinsicParams,
        rightCameraIntrinsicParams,
        rightToLeftRotationVector,
        rightToLeftTranslationVector
        );


    if (pointsIn3D.size() != pointPairs.size())
    {
      std::ostringstream oss;
      oss << "Could not triangulate all points. 2D=" << pointPairs.size() << ", whereas 3D=" << pointsIn3D.size() << std::endl;
      throw std::logic_error(oss.str());
    }

    for (unsigned int i = 0; i < pointsIn3D.size(); i++)
    {
      std::cout << "[" << i << "], 2Dl=(" << pointPairs[i].first.x << ", " << pointPairs[i].first.y << "), 2Dr=(" << pointPairs[i].second.x << ", " << pointPairs[i].second.y << "), 3D=" << pointsIn3D[i].x << ", " << pointsIn3D[i].y << ", " << pointsIn3D[i].z <<  std::endl;
    }

    // Tidy up.
    cvReleaseMat(&intrinsicLeft);
    cvReleaseMat(&intrinsicRight);
    cvReleaseMat(&rightToLeftRotation);
    cvReleaseMat(&rightToLeftTranslation);
  }
  catch(std::logic_error& e)
  {
    std::cerr << "Triangulate2DPointPairsTo3D::Project: exception thrown e=" << e.what() << std::endl;
  }

  // Read in pairs of points.

  return isSuccessful;
}

} // end namespace

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
#include <mitkPointSet.h>
#include <mitkIOUtil.h>

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
                                              const std::string& rightToLeftExtrinsics,
                                              const std::string& outputFileName
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

    std::vector< std::pair<cv::Point2d, cv::Point2d> > pointPairs;
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
      cv::Point2d leftPoint;
      cv::Point2d rightPoint;

      leftPoint.x = numbersOnLine[0];
      leftPoint.y = numbersOnLine[1];
      rightPoint.x = numbersOnLine[2];
      rightPoint.y = numbersOnLine[3];

      pointPairs.push_back(std::pair<cv::Point2d, cv::Point2d>(leftPoint, rightPoint));
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
    std::vector< cv::Point3d > pointsIn3D;
    mitk::PointSet::Pointer ps = mitk::PointSet::New();

    for (unsigned int i = 0; i < pointPairs.size(); i++)
    {
      cv::Point3d pointIn3D = mitk::TriangulatePointPairUsingGeometry(
          pointPairs[i],
          leftIntrinsic,
          rightIntrinsic,
          rightToLeftRotationMatrix,
          rightToLeftTranslationVector
          );
      pointsIn3D.push_back(pointIn3D);

      mitk::Point3D p;
      p[0] = pointsIn3D[i].x;
      p[1] = pointsIn3D[i].y;
      p[2] = pointsIn3D[i].z;
      ps->InsertPoint(i, p);

    }
    mitk::IOUtil::SavePointSet(ps, outputFileName);
    isSuccessful = true;
  }
  catch(std::logic_error& e)
  {
    std::cerr << "Triangulate2DPointPairsTo3D::Project: exception thrown e=" << e.what() << std::endl;
  }

  return isSuccessful;
}

} // end namespace

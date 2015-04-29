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
: m_LeftMaskFileName("")
, m_RightMaskFileName("")
, m_LeftLensToWorldFileName("")
, m_Input2DPointPairsFileName("")
, m_IntrinsicLeftFileName("")
, m_IntrinsicRightFileName("")
, m_RightToLeftExtrinsics("")
, m_OutputFileName("")
, m_BlankValue(0)
{
}


//-----------------------------------------------------------------------------
Triangulate2DPointPairsTo3D::~Triangulate2DPointPairsTo3D()
{
}

//-----------------------------------------------------------------------------
bool Triangulate2DPointPairsTo3D::Triangulate()
{
  bool isSuccessful = false;

  try
  {
    // Load the pairs of 2D points.
    std::ifstream reader(m_Input2DPointPairsFileName.c_str());
    if (!reader)
    {
      std::cerr << "Failed to open " << m_Input2DPointPairsFileName << std::endl;
      return false;
    }
    else
    {
      std::cout << "Opened " << m_Input2DPointPairsFileName << std::endl;
    }

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

      m_PointPairs.push_back(std::pair<cv::Point2d, cv::Point2d>(leftPoint, rightPoint));
    }
    reader.close();

    if (m_PointPairs.size() == 0)
    {
      throw std::logic_error("Failed to read 3D points");
    }

    std::cout << "Triangulate2DPointPairsTo3D: Read in " << m_PointPairs.size() << " point pairs." << std::endl;

    cv::Mat leftIntrinsic = cvCreateMat (3,3,CV_64FC1);
    cv::Mat leftDistortion = cvCreateMat (1,4,CV_64FC1);    // not used (yet)
    cv::Mat rightIntrinsic = cvCreateMat (3,3,CV_64FC1);
    cv::Mat rightDistortion = cvCreateMat (1,4,CV_64FC1);   // not used (yet)
    cv::Mat rightToLeftRotationMatrix = cvCreateMat (3,3,CV_64FC1);
    cv::Mat rightToLeftTranslationVector = cvCreateMat (1,3,CV_64FC1);

    // Load matrices. These throw exceptions if things fail.
    LoadCameraIntrinsicsFromPlainText(m_IntrinsicLeftFileName, &leftIntrinsic, &leftDistortion);
    LoadCameraIntrinsicsFromPlainText(m_IntrinsicRightFileName, &rightIntrinsic, &rightDistortion);
    LoadStereoTransformsFromPlainText(m_RightToLeftExtrinsics, &rightToLeftRotationMatrix, &rightToLeftTranslationVector);

    // batch-triangulate all points.
    std::vector <cv::Point3d> pointsIn3D = TriangulatePointPairsUsingGeometry(
        m_PointPairs,
        leftIntrinsic,
        rightIntrinsic,
        rightToLeftRotationMatrix,
        rightToLeftTranslationVector,
        // choose an arbitrary threshold that is unlikely to overflow.
        std::numeric_limits<int>::max());

    mitk::PointSet::Pointer ps = mitk::PointSet::New();
    for (unsigned int i = 0; i < pointsIn3D.size(); i++)
    {
      mitk::Point3D p;
      p[0] = pointsIn3D[i].x;
      p[1] = pointsIn3D[i].y;
      p[2] = pointsIn3D[i].z;
      ps->InsertPoint(i, p);
    }

    mitk::IOUtil::Save(ps, m_OutputFileName);
    isSuccessful = true;
  }
  catch (const std::logic_error& e)
  {
    std::cerr << "Triangulate2DPointPairsTo3D::Project: exception thrown e=" << e.what() << std::endl;
  }

  return isSuccessful;
}

bool inLeftMask ( const std::pair<cv::Point2d, cv::Point2d>& point )
{
  unsigned int maskValue = 0 ;// mask.at<unsigned int> ( point.first.x, point.first.y );
  if ( maskValue == 0 )
  {
    return true;
  }
  else
  {
    return false;
  }
}
//-----------------------------------------------------------------------------
void Triangulate2DPointPairsTo3D::ApplyMasks()
{
  cv::Mat leftMask;
  cv::Mat rightMask;
  if ( m_LeftMaskFileName != "" )
  {
    leftMask = cv::imread(m_LeftMaskFileName);
    if ( ! leftMask.data )
    {
      MITK_ERROR << "Failed to open " << m_LeftMaskFileName;
    }
  
    m_PointPairs.erase ( std::remove_if ( m_PointPairs.begin(), m_PointPairs.end(), inLeftMask  ), m_PointPairs.end());
/*    std::vector < std::pair<cv::Point2d, cv::Point2d> >::iterator it = m_PointPairs.begin();

    while ( it < m_PointPairs.end() ) 
    {
      unsigned int maskValue = leftMask.at<unsigned int> ( it->first.x, it->first.y );
      if ( maskValue == m_BlankValue ) 
      {
        m_PointPairs.pop(it);
      }
      else;
      it++;
    }
  }*/
  }
}
} // end namespace

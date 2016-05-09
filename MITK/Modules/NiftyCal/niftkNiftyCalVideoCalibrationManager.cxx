/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkNiftyCalVideoCalibrationManager.h"
#include <mitkExceptionMacro.h>
#include <niftkNiftyCalTypes.h>
#include <niftkImageConversion.h>
#include <niftkOpenCVChessboardPointDetector.h>
#include <cv.h>

namespace niftk
{

//-----------------------------------------------------------------------------
NiftyCalVideoCalibrationManager::NiftyCalVideoCalibrationManager()
: m_DataStorage(nullptr)
, m_LeftImageNode(nullptr)
, m_RightImageNode(nullptr)
, m_TrackingTransformNode(nullptr)
, m_MinimumNumberOfSnapshotsForCalibrating(5)
{
}


//-----------------------------------------------------------------------------
NiftyCalVideoCalibrationManager::~NiftyCalVideoCalibrationManager()
{

}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::SetDataStorage(
    const mitk::DataStorage::Pointer& storage)
{
  if (storage.IsNull())
  {
    mitkThrow() << "Null DataStorage passed";
  }
  m_DataStorage = storage;
}


//-----------------------------------------------------------------------------
unsigned int NiftyCalVideoCalibrationManager::GetNumberOfSnapshots() const
{
  return m_LeftPoints.size();
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::Restart()
{
  m_LeftPoints.clear();
  m_RightPoints.clear();
  MITK_INFO << "Restart. Left point size now:" << m_LeftPoints.size() << ", right: " <<  m_RightPoints.size();
}


//-----------------------------------------------------------------------------
bool NiftyCalVideoCalibrationManager::Grab()
{
  bool result = false;

  // To Do: Pause DataSources update - (from plugin layer).

  if (m_LeftImageNode.IsNull())
  {
    mitkThrow() << "Left image should never be NULL";
  }

  bool gotLeft = false;
  bool gotRight = false;

  mitk::Image::Pointer leftImage = dynamic_cast<mitk::Image*>(m_LeftImageNode->GetData());
  if (leftImage.IsNotNull())
  {
    cv::Mat image = niftk::MitkImageToOpenCVMat(leftImage);

    cv::Point2d scaleFactors;
    scaleFactors.x = 1;
    scaleFactors.y = 1;

    cv::Mat greyImage;
    cv::cvtColor(image, greyImage, CV_BGR2GRAY);

    cv::Size2i internalCorners(14, 10);
    niftk::OpenCVChessboardPointDetector detector(internalCorners);
    detector.SetImage(&greyImage);
    detector.SetImageScaleFactor(scaleFactors);

    niftk::PointSet points = detector.GetPoints();
    if (points.size() > 0)
    {
      gotLeft = true;
      m_LeftPoints.push_back(points);
    }
    else
    {
      MITK_INFO << "Failed to extract left camera points.";
    }
  }

  if (m_RightImageNode.IsNotNull())
  {
    mitk::Image::Pointer rightImage = dynamic_cast<mitk::Image*>(m_RightImageNode->GetData());
    if (rightImage.IsNotNull())
    {
      cv::Mat image = niftk::MitkImageToOpenCVMat(rightImage);

      cv::Point2d scaleFactors;
      scaleFactors.x = 1;
      scaleFactors.y = 1;

      cv::Mat greyImage;
      cv::cvtColor(image, greyImage, CV_BGR2GRAY);

      cv::Size2i internalCorners(14, 10);
      niftk::OpenCVChessboardPointDetector detector(internalCorners);
      detector.SetImage(&greyImage);
      detector.SetImageScaleFactor(scaleFactors);

      niftk::PointSet points = detector.GetPoints();
      if (points.size() > 0)
      {
        gotRight = true;
        m_RightPoints.push_back(points);
      }
      else
      {
        MITK_INFO << "Failed to extract right camera points.";
      }
    }
  }

  if (gotLeft &&
      ((m_RightImageNode.IsNotNull() && gotRight) || m_RightImageNode.IsNull())
      )
  {
    result = true;
  }

  MITK_INFO << "Grabbed. Left point size now:" << m_LeftPoints.size() << ", right: " <<  m_RightPoints.size();
  return result;
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::UnGrab()
{
  if (!m_LeftPoints.empty())
  {
    m_LeftPoints.pop_back();
  }

  if (!m_RightPoints.empty())
  {
    m_RightPoints.pop_back();
  }

  MITK_INFO << "UnGrab. Left point size now:" << m_LeftPoints.size() << ", right: " <<  m_RightPoints.size();
}


//-----------------------------------------------------------------------------
double NiftyCalVideoCalibrationManager::Calibrate()
{
  MITK_INFO << "Calibrating.";
  int j=0;
  for (int i = 0; i < 1000000000; i++)
  {
    j++;
  }
  MITK_INFO << "Calibrating - DONE";
  return 2;
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::Save(const std::string dirName)
{
  MITK_INFO << "Saving calibration to " << dirName;
  niftk::NiftyCalTimeType time;
}

} // end namespace

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

const bool                NiftyCalVideoCalibrationManager::DefaultDoIterative(false);
const unsigned int        NiftyCalVideoCalibrationManager::DefaultMinimumNumberOfSnapshotsForCalibrating(3);
const double              NiftyCalVideoCalibrationManager::DefaultScaleFactorX(1);
const double              NiftyCalVideoCalibrationManager::DefaultScaleFactorY(1);
const int                 NiftyCalVideoCalibrationManager::DefaultGridSizeX(14);
const int                 NiftyCalVideoCalibrationManager::DefaultGridSizeY(10);
const std::string         NiftyCalVideoCalibrationManager::DefaultTagFamily("25h7");
const NiftyCalVideoCalibrationManager::CalibrationPatterns NiftyCalVideoCalibrationManager::DefaultCalibrationPattern(NiftyCalVideoCalibrationManager::CHESSBOARD);
const NiftyCalVideoCalibrationManager::HandEyeMethod       NiftyCalVideoCalibrationManager::DefaultHandEyeMethod(NiftyCalVideoCalibrationManager::TSAI);

//-----------------------------------------------------------------------------
NiftyCalVideoCalibrationManager::NiftyCalVideoCalibrationManager()
: m_DataStorage(nullptr)
, m_TrackingTransformNode(nullptr)
, m_DoIterative(NiftyCalVideoCalibrationManager::DefaultDoIterative)
, m_MinimumNumberOfSnapshotsForCalibrating(NiftyCalVideoCalibrationManager::DefaultMinimumNumberOfSnapshotsForCalibrating)
, m_ScaleFactorX(NiftyCalVideoCalibrationManager::DefaultScaleFactorX)
, m_ScaleFactorY(NiftyCalVideoCalibrationManager::DefaultScaleFactorY)
, m_GridSizeX(NiftyCalVideoCalibrationManager::DefaultGridSizeX)
, m_GridSizeY(NiftyCalVideoCalibrationManager::DefaultGridSizeY)
, m_CalibrationPattern(NiftyCalVideoCalibrationManager::DefaultCalibrationPattern)
, m_HandeyeMethod(NiftyCalVideoCalibrationManager::DefaultHandEyeMethod)
, m_TagFamily(NiftyCalVideoCalibrationManager::DefaultTagFamily)
{
  m_ImageNode[0] = nullptr;
  m_ImageNode[1] = nullptr;
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
void NiftyCalVideoCalibrationManager::SetLeftImageNode(mitk::DataNode::Pointer node)
{
  this->m_ImageNode[0] = node;
  this->Modified();
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer NiftyCalVideoCalibrationManager::GetLeftImageNode() const
{
  return m_ImageNode[0];
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::SetRightImageNode(mitk::DataNode::Pointer node)
{
  this->m_ImageNode[1] = node;
  this->Modified();
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer NiftyCalVideoCalibrationManager::GetRightImageNode() const
{
  return m_ImageNode[1];
}


//-----------------------------------------------------------------------------
unsigned int NiftyCalVideoCalibrationManager::GetNumberOfSnapshots() const
{
  return m_Points[0].size();
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::Restart()
{
  for (int i = 0; i < 2; i++)
  {
    m_Points[i].clear();
    m_OriginalImages[i].clear();
    m_ImagesForWarping[i].clear();
  }
  MITK_INFO << "Restart. Left point size now:" << m_Points[0].size() << ", right: " <<  m_Points[1].size();
}


//-----------------------------------------------------------------------------
bool NiftyCalVideoCalibrationManager::ConvertImage(
    mitk::DataNode::Pointer imageNode, cv::Mat& outputImage)
{
  bool converted = false;

  mitk::Image::Pointer inputImage = dynamic_cast<mitk::Image*>(imageNode->GetData());
  if (inputImage.IsNotNull())
  {
    cv::Mat image = niftk::MitkImageToOpenCVMat(inputImage);
    cv::cvtColor(image, outputImage, CV_BGR2GRAY);
    converted = true;
  }

  return converted;
}


//-----------------------------------------------------------------------------
bool NiftyCalVideoCalibrationManager::ExtractPoints(int imageIndex, const cv::Mat& image)
{
  bool result = false;

  cv::Point2d scaleFactors;
  scaleFactors.x = m_ScaleFactorX;
  scaleFactors.y = m_ScaleFactorY;

  cv::Mat copyOfImage1 = image.clone(); // Remember OpenCV reference counting.
  cv::Mat copyOfImage2 = image.clone(); // Remember OpenCV reference counting.

  if (m_CalibrationPattern == CHESSBOARD)
  {
    cv::Size2i internalCorners(m_GridSizeX, m_GridSizeY);

    niftk::OpenCVChessboardPointDetector *openCVDetector1 = new niftk::OpenCVChessboardPointDetector(internalCorners);
    openCVDetector1->SetImageScaleFactor(scaleFactors);
    openCVDetector1->SetImage(&copyOfImage1);

    niftk::PointSet points = openCVDetector1->GetPoints();
    if (points.size() == m_GridSizeX * m_GridSizeY)
    {
      result = true;
      m_Points[imageIndex].push_back(points);

      std::shared_ptr<niftk::IPoint2DDetector> originalDetector(openCVDetector1);
      m_OriginalImages[imageIndex].push_back(std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(originalDetector, copyOfImage1));
      dynamic_cast<niftk::OpenCVChessboardPointDetector*>(m_OriginalImages[imageIndex].back().first.get())->SetImage(&(m_OriginalImages[imageIndex].back().second));

      niftk::OpenCVChessboardPointDetector *openCVDetector2 = new niftk::OpenCVChessboardPointDetector(internalCorners);
      openCVDetector2->SetImageScaleFactor(scaleFactors);
      openCVDetector2->SetImage(&copyOfImage2);

      std::shared_ptr<niftk::IPoint2DDetector> warpedDetector(openCVDetector2);
      m_ImagesForWarping[imageIndex].push_back(std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(warpedDetector, copyOfImage2));
      dynamic_cast<niftk::OpenCVChessboardPointDetector*>(m_ImagesForWarping[imageIndex].back().first.get())->SetImage(&(m_ImagesForWarping[imageIndex].back().second));
    }
  }
  else if (m_CalibrationPattern == CIRCLE_GRID)
  {
    std::cerr << "Matt, m_CalibrationPattern == CIRCLE_GRID" << std::endl;
  }
  else if (m_CalibrationPattern == APRIL_TAGS)
  {
    std::cerr << "Matt, m_CalibrationPattern == APRIL_TAGS" << std::endl;
  }
  else
  {
    mitkThrow() << "Invalid calibration pattern.";
  }

  return result;
}


//-----------------------------------------------------------------------------
bool NiftyCalVideoCalibrationManager::Grab()
{
  bool result = false;

  if (m_ImageNode[0].IsNull())
  {
    mitkThrow() << "Left image should never be NULL";
  }

  bool converted[2] = {false, false};
  bool extracted[2] = {false, false};

  // Its only a loop over 2 channels, but then we can use OpenMP!
  for (int i = 0; i < 2; i++)
  {
    if (m_ImageNode[i].IsNotNull())
    {
      converted[i] = this->ConvertImage(m_ImageNode[i], m_TmpImage[i]);
      if (converted[i])
      {
        extracted[i] = this->ExtractPoints(i, m_TmpImage[i]);
      }
    }
  }

  if (converted[0] && extracted[0] // must do left.
      && ((m_ImageNode[1].IsNotNull() && converted[1] && extracted[1])
          || m_ImageNode[1].IsNull())
      )
  {
    result = true;
  }

  MITK_INFO << "Grabbed. Left point size now:" << m_Points[0].size() << ", right: " <<  m_Points[1].size();
  return result;
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::UnGrab()
{
  for (int i = 0; i < 2; i++)
  {
    if (!m_Points[i].empty())
    {
      m_Points[i].pop_back();
      m_OriginalImages[i].pop_back();
      m_ImagesForWarping[i].pop_back();
    }
  }

  MITK_INFO << "UnGrab. Left point size now:" << m_Points[0].size() << ", right: " <<  m_Points[1].size();
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
}

} // end namespace

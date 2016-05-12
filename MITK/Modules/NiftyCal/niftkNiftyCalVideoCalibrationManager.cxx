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
#include <mitkCoordinateAxesData.h>
#include <niftkNiftyCalTypes.h>
#include <niftkImageConversion.h>
#include <niftkOpenCVChessboardPointDetector.h>
#include <niftkOpenCVCirclesPointDetector.h>
#include <niftkAprilTagsPointDetector.h>
#include <niftkIOUtilities.h>
#include <niftkMonoCameraCalibration.h>
#include <niftkStereoCameraCalibration.h>
#include <niftkIterativeMonoCameraCalibration.h>
#include <niftkIterativeStereoCameraCalibration.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>

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
void NiftyCalVideoCalibrationManager::Set3DModelFileName(const std::string fileName)
{
  if (fileName.empty())
  {
    mitkThrow() << "Empty 3D model file name";
  }

  m_3DModelFileName = fileName;
  niftk::Model3D model = niftk::LoadModel3D(m_3DModelFileName);

  if (model.empty())
  {
    mitkThrow() << "Failed to load model points";
  }

  m_3DModelPoints = model;
  this->Modified();
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
    m_TrackingMatrices.clear();
  }
  MITK_INFO << "Restart. Left point size now:" << m_Points[0].size() << ", right: " <<  m_Points[1].size();
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::ConvertImage(
    mitk::DataNode::Pointer imageNode, cv::Mat& outputImage)
{
  mitk::Image::Pointer inputImage = dynamic_cast<mitk::Image*>(imageNode->GetData());

  if (inputImage.IsNull())
  {
    mitkThrow() << "Null input image.";
  }

  cv::Mat image = niftk::MitkImageToOpenCVMat(inputImage);
  cv::cvtColor(image, outputImage, CV_BGR2GRAY);
  m_ImageSize.width = image.cols;
  m_ImageSize.height = image.rows;
}


//-----------------------------------------------------------------------------
bool NiftyCalVideoCalibrationManager::ExtractPoints(int imageIndex, const cv::Mat& image)
{
  bool isSuccessful = false;

  cv::Point2d scaleFactors;
  scaleFactors.x = m_ScaleFactorX;
  scaleFactors.y = m_ScaleFactorY;

  cv::Mat copyOfImage1 = image.clone(); // Remember OpenCV reference counting.
  cv::Mat copyOfImage2 = image.clone(); // Remember OpenCV reference counting.

  // Watch out: OpenCV reference counts the image data block.
  // So, if you create two cv::Mat, using say the copy constructor
  // or assignment constructor, both cv::Mat point to the same memory
  // block, unless you explicitly call the clone method. This
  // causes a problem when we store them in a STL container.
  // So, in the code below, we add the detector and the image into
  // a std::pair, and stuff it in a list. This causes the cv::Mat
  // in the list to be a different container object to the one you
  // started with, which is why we have to dynamic cast, and then
  // call SetImage again.

  if (m_CalibrationPattern == CHESSBOARD)
  {
    cv::Size2i internalCorners(m_GridSizeX, m_GridSizeY);

    niftk::OpenCVChessboardPointDetector *openCVDetector1 = new niftk::OpenCVChessboardPointDetector(internalCorners);
    openCVDetector1->SetImageScaleFactor(scaleFactors);
    openCVDetector1->SetImage(&copyOfImage1);

    niftk::PointSet points = openCVDetector1->GetPoints();
    if (points.size() == m_GridSizeX * m_GridSizeY)
    {
      isSuccessful = true;
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
    cv::Size2i internalCorners(m_GridSizeX, m_GridSizeY);

    niftk::OpenCVCirclesPointDetector *openCVDetector1 = new niftk::OpenCVCirclesPointDetector(internalCorners);
    openCVDetector1->SetImageScaleFactor(scaleFactors);
    openCVDetector1->SetImage(&copyOfImage1);

    niftk::PointSet points = openCVDetector1->GetPoints();
    if (points.size() == m_GridSizeX * m_GridSizeY)
    {
      isSuccessful = true;
      m_Points[imageIndex].push_back(points);

      std::shared_ptr<niftk::IPoint2DDetector> originalDetector(openCVDetector1);
      m_OriginalImages[imageIndex].push_back(std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(originalDetector, copyOfImage1));
      dynamic_cast<niftk::OpenCVCirclesPointDetector*>(m_OriginalImages[imageIndex].back().first.get())->SetImage(&(m_OriginalImages[imageIndex].back().second));

      niftk::OpenCVCirclesPointDetector *openCVDetector2 = new niftk::OpenCVCirclesPointDetector(internalCorners);
      openCVDetector2->SetImageScaleFactor(scaleFactors);
      openCVDetector2->SetImage(&copyOfImage2);

      std::shared_ptr<niftk::IPoint2DDetector> warpedDetector(openCVDetector2);
      m_ImagesForWarping[imageIndex].push_back(std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(warpedDetector, copyOfImage2));
      dynamic_cast<niftk::OpenCVCirclesPointDetector*>(m_ImagesForWarping[imageIndex].back().first.get())->SetImage(&(m_ImagesForWarping[imageIndex].back().second));
    }
  }
  else if (m_CalibrationPattern == APRIL_TAGS)
  {
    niftk::AprilTagsPointDetector *openCVDetector1 = new niftk::AprilTagsPointDetector(true, m_TagFamily, 0, 0.8);
    openCVDetector1->SetImageScaleFactor(scaleFactors);
    openCVDetector1->SetImage(&copyOfImage1);

    niftk::PointSet points = openCVDetector1->GetPoints();
    if (points.size() > 0)
    {
      isSuccessful = true;
      m_Points[imageIndex].push_back(points);

      std::shared_ptr<niftk::IPoint2DDetector> originalDetector(openCVDetector1);
      m_OriginalImages[imageIndex].push_back(std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(originalDetector, copyOfImage1));
      dynamic_cast<niftk::AprilTagsPointDetector*>(m_OriginalImages[imageIndex].back().first.get())->SetImage(&(m_OriginalImages[imageIndex].back().second));

      niftk::AprilTagsPointDetector *openCVDetector2 = new niftk::AprilTagsPointDetector(true, m_TagFamily, 0, 0.8);
      openCVDetector2->SetImageScaleFactor(scaleFactors);
      openCVDetector2->SetImage(&copyOfImage2);

      std::shared_ptr<niftk::IPoint2DDetector> warpedDetector(openCVDetector2);
      m_ImagesForWarping[imageIndex].push_back(std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(warpedDetector, copyOfImage2));
      dynamic_cast<niftk::AprilTagsPointDetector*>(m_ImagesForWarping[imageIndex].back().first.get())->SetImage(&(m_ImagesForWarping[imageIndex].back().second));
    }
  }
  else
  {
    mitkThrow() << "Invalid calibration pattern.";
  }

  return isSuccessful;
}


//-----------------------------------------------------------------------------
bool NiftyCalVideoCalibrationManager::Grab()
{
  bool isSuccessful = false;

  if (m_ImageNode[0].IsNull())
  {
    mitkThrow() << "Left image should never be NULL";
  }

  // 3 entries - first two represent image nodes, third represents the tracker node.
  bool extracted[3] = {false, false, false};

  // Deliberately looping over only the entries for two image nodes.
  for (int i = 0; i < 2; i++)
  {
    if (m_ImageNode[i].IsNotNull())
    {
      this->ConvertImage(m_ImageNode[i], m_TmpImage[i]);
      extracted[i] = this->ExtractPoints(i, m_TmpImage[i]);
    }
  }

  // Now we extract the tracking node.
  if (m_TrackingTransformNode.IsNotNull())
  {
    mitk::CoordinateAxesData::Pointer tracking = dynamic_cast<mitk::CoordinateAxesData*>(m_TrackingTransformNode->GetData());
    if (tracking.IsNull())
    {
      mitkThrow() << "Tracking node contains null tracking matrix";
    }
    vtkSmartPointer<vtkMatrix4x4> mat = vtkSmartPointer<vtkMatrix4x4>::New();
    tracking->GetVtkMatrix(*mat);

    cv::Matx44d openCVMat;

    for (int r = 0; r < 4; r++)
    {
      for (int c = 0; c < 4; c++)
      {
        openCVMat(r,c) = mat->GetElement(r,c);
      }
    }
    m_TrackingMatrices.push_back(openCVMat);
  }

  // Then we check if we got everything, and therefore we are successful.
  if (extracted[0] // must always do left.
      && ((m_ImageNode[1].IsNotNull() && extracted[1]) // right image is optional
          || m_ImageNode[1].IsNull())
      && ((m_TrackingTransformNode.IsNotNull() && extracted[2]) // tracking node is optional
          || m_TrackingTransformNode.IsNull())
      )
  {
    isSuccessful = true;
  }

  MITK_INFO << "Grabbed. Left point size now:" << m_Points[0].size() << ", right: " <<  m_Points[1].size();
  return isSuccessful;
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
      m_TrackingMatrices.pop_back();
    }
  }

  MITK_INFO << "UnGrab. Left point size now:" << m_Points[0].size() << ", right: " <<  m_Points[1].size();
}


//-----------------------------------------------------------------------------
double NiftyCalVideoCalibrationManager::Calibrate()
{
  MITK_INFO << "Calibrating.";

  double rms = 0;

  if (m_ImageNode[0].IsNull())
  {
    mitkThrow() << "Left image should never be NULL.";
  }

  if (m_3DModelPoints.empty())
  {
    mitkThrow() << "Model should never be empty.";
  }

  if (m_DoIterative)
  {
    rms = niftk::IterativeMonoCameraCalibration(
          m_3DModelPoints,
          m_ReferenceDataForIterativeCalib,
          m_OriginalImages[0],
          m_ImagesForWarping[0],
          m_ImageSize,
          m_Intrinsic[0],
          m_Distortion[0],
          m_Rvecs[0],
          m_Tvecs[0]
          );

    if (m_ImageNode[1].IsNotNull())
    {
      rms = niftk::IterativeStereoCameraCalibration(
            m_3DModelPoints,
            m_ReferenceDataForIterativeCalib,
            m_OriginalImages[0],
            m_OriginalImages[1],
            m_ImageSize,
            m_ImagesForWarping[0],
            m_Intrinsic[0],
            m_Distortion[0],
            m_Rvecs[0],
            m_Tvecs[0],
            m_ImagesForWarping[1],
            m_Intrinsic[1],
            m_Distortion[1],
            m_Rvecs[1],
            m_Tvecs[1],
            m_EssentialMatrix,
            m_FundamentalMatrix,
            m_Left2RightRotation,
            m_Left2RightTranslation
            );
    }
  }
  else
  {
    rms = niftk::MonoCameraCalibration(
          m_3DModelPoints,
          m_Points[0],
          m_ImageSize,
          m_Intrinsic[0],
          m_Distortion[0],
          m_Rvecs[0],
          m_Tvecs[0]
          );

    if (m_ImageNode[1].IsNotNull())
    {

      niftk::MonoCameraCalibration(
            m_3DModelPoints,
            m_Points[1],
            m_ImageSize,
            m_Intrinsic[1],
            m_Distortion[1],
            m_Rvecs[1],
            m_Tvecs[1]
            );

      rms = niftk::StereoCameraCalibration(
            m_3DModelPoints,
            m_Points[0],
            m_Points[1],
            m_ImageSize,
            m_Intrinsic[0],
            m_Distortion[0],
            m_Rvecs[0],
            m_Tvecs[0],
            m_Intrinsic[1],
            m_Distortion[1],
            m_Rvecs[1],
            m_Tvecs[1],
            m_EssentialMatrix,
            m_FundamentalMatrix,
            m_Left2RightRotation,
            m_Left2RightTranslation,
            CV_CALIB_USE_INTRINSIC_GUESS
            );
    }
  }

  // To Do: Hand-Eye methods
  if (m_HandeyeMethod == TSAI)
  {
  }
  else if (m_HandeyeMethod == DIRECT)
  {
  }
  else if (m_HandeyeMethod == MALTI)
  {
  }

  MITK_INFO << "Calibrating - DONE.";
  return rms;
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::Save(const std::string dirName)
{
  MITK_INFO << "Saving calibration to " << dirName << ".";

  MITK_INFO << "Saving calibration to " << dirName << " - DONE.";
}

} // end namespace

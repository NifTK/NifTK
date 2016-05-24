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
#include <mitkOpenCVMaths.h>
#include <niftkFileHelper.h>
#include <niftkImageConversion.h>

#include <niftkMatrixUtilities.h>
#include <niftkNiftyCalTypes.h>
#include <niftkOpenCVChessboardPointDetector.h>
#include <niftkOpenCVCirclesPointDetector.h>
#include <niftkAprilTagsPointDetector.h>
#include <niftkIOUtilities.h>
#include <niftkMonoCameraCalibration.h>
#include <niftkStereoCameraCalibration.h>
#include <niftkIterativeMonoCameraCalibration.h>
#include <niftkIterativeStereoCameraCalibration.h>
#include <niftkNonLinearMaltiHandEyeOptimiser.h>
#include <niftkNonLinearMaltiNDOFHandEyeOptimiser.h>
#include <niftkNonLinearMaltiStereoHandEyeOptimiser.h>

#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>
#include <highgui.h>
#include <sstream>

namespace niftk
{

const bool                NiftyCalVideoCalibrationManager::DefaultDoIterative(false);
const unsigned int        NiftyCalVideoCalibrationManager::DefaultMinimumNumberOfSnapshotsForCalibrating(5);
const double              NiftyCalVideoCalibrationManager::DefaultScaleFactorX(1);
const double              NiftyCalVideoCalibrationManager::DefaultScaleFactorY(1);
const int                 NiftyCalVideoCalibrationManager::DefaultGridSizeX(14);
const int                 NiftyCalVideoCalibrationManager::DefaultGridSizeY(10);
const std::string         NiftyCalVideoCalibrationManager::DefaultTagFamily("25h7");

const NiftyCalVideoCalibrationManager::CalibrationPatterns
  NiftyCalVideoCalibrationManager::DefaultCalibrationPattern(NiftyCalVideoCalibrationManager::CHESS_BOARD);

const NiftyCalVideoCalibrationManager::HandEyeMethod
  NiftyCalVideoCalibrationManager::DefaultHandEyeMethod(NiftyCalVideoCalibrationManager::TSAI_1989);

//-----------------------------------------------------------------------------
NiftyCalVideoCalibrationManager::NiftyCalVideoCalibrationManager()
: m_DataStorage(nullptr)
, m_TrackingTransformNode(nullptr)
, m_ReferenceTrackingTransformNode(nullptr)
, m_DoIterative(NiftyCalVideoCalibrationManager::DefaultDoIterative)
, m_MinimumNumberOfSnapshotsForCalibrating(
    NiftyCalVideoCalibrationManager::DefaultMinimumNumberOfSnapshotsForCalibrating)
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
  for (int i = 0; i < 4; i++)                           // 4 different methods - see HandEyeMethod enum
  {
    m_HandEyeMatrices[0].push_back(cv::Matx44d::eye()); // left
    m_HandEyeMatrices[1].push_back(cv::Matx44d::eye()); // right
    m_ReferenceHandEyeMatrices[0].push_back(cv::Matx44d::eye()); // left
    m_ReferenceHandEyeMatrices[1].push_back(cv::Matx44d::eye()); // right
  }
  m_ModelPointsToVisualise = mitk::PointSet::New();
  m_ModelPointsToVisualiseDataNode = mitk::DataNode::New();
  m_ModelPointsToVisualiseDataNode->SetData(m_ModelPointsToVisualise);
  m_ModelPointsToVisualiseDataNode->SetName("CalibrationModelPoints");
  m_ModelPointsToVisualiseDataNode->SetBoolProperty("helper object", true);
}


//-----------------------------------------------------------------------------
NiftyCalVideoCalibrationManager::~NiftyCalVideoCalibrationManager()
{
  if (m_DataStorage.IsNotNull())
  {
    if (m_DataStorage->Exists(m_ModelPointsToVisualiseDataNode))
    {
      m_DataStorage->Remove(m_ModelPointsToVisualiseDataNode);
    }
  }
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::SetDataStorage(
    const mitk::DataStorage::Pointer storage)
{
  if (storage.IsNull())
  {
    mitkThrow() << "Null DataStorage passed.";
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
void NiftyCalVideoCalibrationManager::SetModelFileName(const std::string& fileName)
{
  if (fileName.empty())
  {
    mitkThrow() << "Empty 3D model file name.";
  }

  niftk::Model3D model = niftk::LoadModel3D(fileName);
  if (model.empty())
  {
    mitkThrow() << "Failed to load model points.";
  }

  m_ModelFileName = fileName;
  m_ModelPoints = model;

  m_ModelPointsToVisualise->Clear();
  niftk::Model3D::const_iterator iter;
  for (iter = m_ModelPoints.begin();
       iter != m_ModelPoints.end();
       ++iter
       )
  {
    cv::Point3d p1 = (*iter).second.point;
    mitk::Point3D p2;
    p2[0] = p1.x;
    p2[1] = p1.y;
    p2[2] = p1.z;
    m_ModelPointsToVisualise->InsertPoint((*iter).first, p2);
  }
  if (m_DataStorage.IsNotNull() && !m_DataStorage->Exists(m_ModelPointsToVisualiseDataNode))
  {
    m_DataStorage->Add(m_ModelPointsToVisualiseDataNode);
  }
  this->Modified();
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::SetReferenceDataFileNames(
    const std::string& imageFileName, const std::string& pointsFileName)
{
  if (!imageFileName.empty() && !pointsFileName.empty())
  {
    cv::Mat referenceImage = cv::imread(imageFileName);
    if (referenceImage.rows == 0 || referenceImage.cols == 0)
    {
      mitkThrow() << "Failed to read reference image:" << imageFileName;
    }

    cv::Mat referenceImageGreyScale;
    cv::cvtColor(referenceImage, referenceImageGreyScale, CV_BGR2GRAY);

    std::pair< cv::Mat, niftk::PointSet> referenceImageData;
    referenceImageData.first = referenceImageGreyScale;
    referenceImageData.second = niftk::LoadPointSet(pointsFileName);

    if (referenceImageData.second.size() == 0)
    {
      mitkThrow() << "Failed to read reference points:" << pointsFileName;
    }

    m_ReferenceImageFileName = imageFileName;
    m_ReferencePointsFileName = pointsFileName;
    m_ReferenceDataForIterativeCalib = referenceImageData;
    this->Modified();
  }
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::SetModelToTrackerFileName(
    const std::string& fileName)
{
  if (!fileName.empty())
  {
    m_ModelToTracker = niftk::LoadMatrix(fileName);
    this->Modified();
  }
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::SetOutputDirName(const std::string& dirName)
{
  if (dirName.empty())
  {
    mitkThrow() << "Empty output directory name.";
  }

  m_OutputDirName = (dirName + niftk::GetFileSeparator());
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
    m_ReferenceTrackingMatrices.clear();
  }
  MITK_INFO << "Restart. Left point size now:" << m_Points[0].size()
            << ", right: " <<  m_Points[1].size();
}


//-----------------------------------------------------------------------------
std::list<cv::Matx44d> NiftyCalVideoCalibrationManager::ExtractCameraMatrices(int imageIndex)
{
  std::list<cv::Matx44d> cameraMatrices;
  for (int i = 0; i < m_Rvecs[imageIndex].size(); i++)
  {
    cv::Matx44d mat = niftk::RodriguesToMatrix(m_Rvecs[imageIndex][i],
                                               m_Tvecs[imageIndex][i]
                                               );
    cameraMatrices.push_back(mat);
  }
  return cameraMatrices;
}


//-----------------------------------------------------------------------------
std::list<cv::Matx44d> NiftyCalVideoCalibrationManager::ExtractTrackingMatrices(bool useReference)
{
  std::list<cv::Matx44d> trackingMatrices;

  if (useReference)
  {
    if (m_TrackingMatrices.size() != m_ReferenceTrackingMatrices.size())
    {
      mitkThrow() << "Different number of tracking and reference matrices";
    }

    std::list<cv::Matx44d>::const_iterator trackingIter;
    std::list<cv::Matx44d>::const_iterator referenceIter;
    for (trackingIter = m_TrackingMatrices.begin(),
         referenceIter = m_ReferenceTrackingMatrices.begin();
         trackingIter != m_TrackingMatrices.end() && referenceIter != m_ReferenceTrackingMatrices.end();
         ++trackingIter,
         ++referenceIter
        )
    {
      trackingMatrices.push_back((*trackingIter) * ((*referenceIter).inv()));
    }
  }
  else
  {
    std::list<cv::Matx44d>::const_iterator trackingIter;
    for (trackingIter = m_TrackingMatrices.begin();
         trackingIter != m_TrackingMatrices.end();
         ++trackingIter
         )
    {
      trackingMatrices.push_back(*trackingIter);
    }
  }

  return trackingMatrices;
}


//-----------------------------------------------------------------------------
std::vector<cv::Mat> NiftyCalVideoCalibrationManager::ConvertMatrices(const std::list<cv::Matx44d>& list)
{
  std::vector<cv::Mat> matrices;
  std::list<cv::Matx44d>::const_iterator iter;

  for (iter = list.begin();
       iter != list.end();
       ++iter
       )
  {
    cv::Mat tmp(*iter);
    matrices.push_back(tmp);
  }
  return matrices;
}


//-----------------------------------------------------------------------------
cv::Matx44d NiftyCalVideoCalibrationManager::DoTsaiHandEye(int imageIndex, bool useReference)
{
  double residualRotation = 0;
  double residualTranslation = 0;

  std::list<cv::Matx44d> cameraMatrices = this->ExtractCameraMatrices(imageIndex);
  std::list<cv::Matx44d> trackingMatrices = this->ExtractTrackingMatrices(useReference);

  cv::Matx44d handEye =
    niftk::CalculateHandEyeUsingTsaisMethod(
      trackingMatrices,
      cameraMatrices,
      residualRotation,
      residualTranslation
      );

  std::cout << "Tsai 1989, linear hand-eye: rot=" << residualRotation
            << ", trans=" << residualTranslation << std::endl;

  return handEye;
}


//-----------------------------------------------------------------------------
cv::Matx44d NiftyCalVideoCalibrationManager::DoKangHandEye(int imageIndex, bool useReference)
{
  std::list<cv::Matx44d> cameraMatrices = this->ExtractCameraMatrices(imageIndex);
  std::list<cv::Matx44d> trackingMatrices = this->ExtractTrackingMatrices(useReference);

  cv::Matx44d handEye =
    niftk::CalculateHandEyeByDirectMatrixMultiplication(
      m_ModelToTracker,
      trackingMatrices,
      cameraMatrices
      );

  return handEye;
}


//-----------------------------------------------------------------------------
cv::Matx44d NiftyCalVideoCalibrationManager::DoMaltiHandEye(int imageIndex, bool useReference)
{
  std::list<cv::Matx44d> cameraMatrices = this->ExtractCameraMatrices(imageIndex);
  std::list<cv::Matx44d> trackingMatrices = this->ExtractTrackingMatrices(useReference);

  // Assumes we have done Tsai first.
  cv::Matx44d handEye = m_HandEyeMatrices[imageIndex][TSAI_1989];
  if (useReference)
  {
    handEye = m_ReferenceHandEyeMatrices[imageIndex][TSAI_1989];
  }

  cv::Matx44d modelToWorld = niftk::CalculateAverageModelToWorld(
        handEye,
        trackingMatrices,
        cameraMatrices
        );

  niftk::NonLinearMaltiHandEyeOptimiser::Pointer optimiser = niftk::NonLinearMaltiHandEyeOptimiser::New();
  optimiser->SetModel(&m_ModelPoints);
  optimiser->SetPoints(&m_Points[imageIndex]);
  optimiser->SetHandMatrices(&trackingMatrices);

  // We clone them, so we dont modify the member variables m_Intrinsic, m_Distortion.
  cv::Mat intrinsic = m_Intrinsic[imageIndex].clone();
  cv::Mat distortion = m_Distortion[imageIndex].clone();

  double reprojectionRMS = optimiser->Optimise(modelToWorld,
                                               handEye,
                                               intrinsic,
                                               distortion
                                              );

  std::cout << "Malti 2013, non-linear hand-eye: rms=" << reprojectionRMS << std::endl;

  return handEye;
}


//-----------------------------------------------------------------------------
cv::Matx44d NiftyCalVideoCalibrationManager::DoFullExtrinsicHandEye(int imageIndex, bool useReference)
{
  std::list<cv::Matx44d> cameraMatrices = this->ExtractCameraMatrices(imageIndex);
  std::list<cv::Matx44d> trackingMatrices = this->ExtractTrackingMatrices(useReference);

  // Assumes we have done Tsai first.
  cv::Matx44d handEye = m_HandEyeMatrices[imageIndex][TSAI_1989];
  if (useReference)
  {
    handEye = m_ReferenceHandEyeMatrices[imageIndex][TSAI_1989];
  }

  cv::Matx44d modelToWorld = niftk::CalculateAverageModelToWorld(
        handEye,
        trackingMatrices,
        cameraMatrices
        );

  niftk::NonLinearMaltiNDOFHandEyeOptimiser::Pointer optimiser = niftk::NonLinearMaltiNDOFHandEyeOptimiser::New();
  optimiser->SetModel(&m_ModelPoints);
  optimiser->SetPoints(&m_Points[imageIndex]);
  optimiser->SetHandMatrices(&trackingMatrices);
  optimiser->SetIntrinsic(&m_Intrinsic[imageIndex]);   // i.e. we DON'T optimise these.
  optimiser->SetDistortion(&m_Distortion[imageIndex]); // i.e. we DON'T optimise these.

  double reprojectionRMS = optimiser->Optimise(modelToWorld, // We do optimise all extrinsic
                                               handEye       // but we don't really care about
                                              );             // the extrinsic results.

  std::cout << "Malti 2013, non-linear hand-eye, but with full extrinsic: "
            << "rms=" << reprojectionRMS << std::endl;

  return handEye;
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::DoFullExtrinsicHandEyeInStereo(cv::Matx44d& leftHandEye,
                                                                     cv::Matx44d& rightHandEye,
                                                                     bool useReference
                                                                     )
{
  std::list<cv::Matx44d> cameraMatrices = this->ExtractCameraMatrices(0);
  std::list<cv::Matx44d> trackingMatrices = this->ExtractTrackingMatrices(useReference);

  // Assumes we have done Tsai first.
  cv::Matx44d handEye = m_HandEyeMatrices[0][TSAI_1989]; // 0 == from left.
  if (useReference)
  {
    handEye = m_ReferenceHandEyeMatrices[0][TSAI_1989];
  }

  cv::Matx44d modelToWorld = niftk::CalculateAverageModelToWorld(
        handEye,
        trackingMatrices,
        cameraMatrices
        );

  niftk::NonLinearMaltiStereoHandEyeOptimiser::Pointer optimiser = niftk::NonLinearMaltiStereoHandEyeOptimiser::New();
  optimiser->SetModel(&m_ModelPoints);
  optimiser->SetPoints(&m_Points[0]);
  optimiser->SetRightHandPoints(&m_Points[1]);
  optimiser->SetHandMatrices(&trackingMatrices);
  optimiser->SetLeftIntrinsic(&m_Intrinsic[0]);    // i.e. we DON'T optimise these.
  optimiser->SetLeftDistortion(&m_Distortion[0]);  // i.e. we DON'T optimise these.
  optimiser->SetRightIntrinsic(&m_Intrinsic[0]);   // i.e. we DON'T optimise these.
  optimiser->SetRightDistortion(&m_Distortion[0]); // i.e. we DON'T optimise these.

  cv::Matx44d stereoExtrinsics = niftk::RodriguesToMatrix(m_RightToLeftRotation, m_RightToLeftTranslation);

  optimiser->Optimise(modelToWorld, handEye, stereoExtrinsics);

  leftHandEye = handEye;
  rightHandEye = (stereoExtrinsics.inv()) * handEye;
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

  if (m_CalibrationPattern == CHESS_BOARD)
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
      m_OriginalImages[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(originalDetector, copyOfImage1));
      dynamic_cast<niftk::OpenCVChessboardPointDetector*>(
        m_OriginalImages[imageIndex].back().first.get())->SetImage(&(m_OriginalImages[imageIndex].back().second));

      niftk::OpenCVChessboardPointDetector *openCVDetector2 =
        new niftk::OpenCVChessboardPointDetector(internalCorners);
      openCVDetector2->SetImageScaleFactor(scaleFactors);
      openCVDetector2->SetImage(&copyOfImage2);

      std::shared_ptr<niftk::IPoint2DDetector> warpedDetector(openCVDetector2);
      m_ImagesForWarping[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(warpedDetector, copyOfImage2));
      dynamic_cast<niftk::OpenCVChessboardPointDetector*>(
        m_ImagesForWarping[imageIndex].back().first.get())->SetImage(&(m_ImagesForWarping[imageIndex].back().second));
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
      m_OriginalImages[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(originalDetector, copyOfImage1));
      dynamic_cast<niftk::OpenCVCirclesPointDetector*>(
        m_OriginalImages[imageIndex].back().first.get())->SetImage(&(m_OriginalImages[imageIndex].back().second));

      niftk::OpenCVCirclesPointDetector *openCVDetector2 =
        new niftk::OpenCVCirclesPointDetector(internalCorners);
      openCVDetector2->SetImageScaleFactor(scaleFactors);
      openCVDetector2->SetImage(&copyOfImage2);

      std::shared_ptr<niftk::IPoint2DDetector> warpedDetector(openCVDetector2);
      m_ImagesForWarping[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(warpedDetector, copyOfImage2));
      dynamic_cast<niftk::OpenCVCirclesPointDetector*>(
        m_ImagesForWarping[imageIndex].back().first.get())->SetImage(&(m_ImagesForWarping[imageIndex].back().second));
    }
  }
  else if (m_CalibrationPattern == APRIL_TAGS)
  {
    niftk::AprilTagsPointDetector *openCVDetector1 =
      new niftk::AprilTagsPointDetector(true, m_TagFamily, 0, 0.8);
    openCVDetector1->SetImageScaleFactor(scaleFactors);
    openCVDetector1->SetImage(&copyOfImage1);

    niftk::PointSet points = openCVDetector1->GetPoints();
    if (points.size() > 0)
    {
      isSuccessful = true;
      m_Points[imageIndex].push_back(points);

      std::shared_ptr<niftk::IPoint2DDetector> originalDetector(openCVDetector1);
      m_OriginalImages[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(originalDetector, copyOfImage1));
      dynamic_cast<niftk::AprilTagsPointDetector*>(
        m_OriginalImages[imageIndex].back().first.get())->SetImage(&(m_OriginalImages[imageIndex].back().second));

      niftk::AprilTagsPointDetector *openCVDetector2 =
        new niftk::AprilTagsPointDetector(true, m_TagFamily, 0, 0.8);
      openCVDetector2->SetImageScaleFactor(scaleFactors);
      openCVDetector2->SetImage(&copyOfImage2);

      std::shared_ptr<niftk::IPoint2DDetector> warpedDetector(openCVDetector2);
      m_ImagesForWarping[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(warpedDetector, copyOfImage2));
      dynamic_cast<niftk::AprilTagsPointDetector*>(
        m_ImagesForWarping[imageIndex].back().first.get())->SetImage(&(m_ImagesForWarping[imageIndex].back().second));
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
    mitkThrow() << "Left image should never be NULL.";
  }

  // 4 entries - 1,2 represent image nodes, 3,4 represents the tracker nodes.
  bool extracted[4] = {false, false, false, false};

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
    mitk::CoordinateAxesData::Pointer tracking = dynamic_cast<mitk::CoordinateAxesData*>(
          m_TrackingTransformNode->GetData());

    if (tracking.IsNull())
    {
      mitkThrow() << "Tracking node contains null tracking matrix.";
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
    extracted[2] = true;
  }

  // Now we extract the reference tracking node - ToDo: fix code duplication.
  if (m_ReferenceTrackingTransformNode.IsNotNull())
  {
    mitk::CoordinateAxesData::Pointer tracking = dynamic_cast<mitk::CoordinateAxesData*>(
          m_ReferenceTrackingTransformNode->GetData());

    if (tracking.IsNull())
    {
      mitkThrow() << "Reference tracking node contains null tracking matrix.";
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
    m_ReferenceTrackingMatrices.push_back(openCVMat);
    extracted[3] = true;
  }

  // Then we check if we got everything, and therefore we are successful.
  if (extracted[0] // must always do left.
      && ((m_ImageNode[1].IsNotNull() && extracted[1]) // right image is optional
          || m_ImageNode[1].IsNull())
      && ((m_TrackingTransformNode.IsNotNull() && extracted[2]) // tracking node is optional
          || m_TrackingTransformNode.IsNull())
      && ((m_ReferenceTrackingTransformNode.IsNotNull() && extracted[3]) // reference tracking node is optional
          || m_ReferenceTrackingTransformNode.IsNull())
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
      m_ReferenceTrackingMatrices.pop_back();
    }
  }

  MITK_INFO << "UnGrab. Left point size now:" << m_Points[0].size() << ", right:" <<  m_Points[1].size();
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

  if (m_ModelPoints.empty())
  {
    mitkThrow() << "Model should never be empty.";
  }

  if (m_DoIterative)
  {
    rms = niftk::IterativeMonoCameraCalibration(
          m_ModelPoints,
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
            m_ModelPoints,
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
            m_RightToLeftRotation,
            m_RightToLeftTranslation
            );
    }
  }
  else
  {
    rms = niftk::MonoCameraCalibration(
          m_ModelPoints,
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
            m_ModelPoints,
            m_Points[1],
            m_ImageSize,
            m_Intrinsic[1],
            m_Distortion[1],
            m_Rvecs[1],
            m_Tvecs[1]
            );

      rms = niftk::StereoCameraCalibration(
            m_ModelPoints,
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
            m_RightToLeftRotation,
            m_RightToLeftTranslation,
            CV_CALIB_USE_INTRINSIC_GUESS
            );
    }
  }

  // Do all hand-eye methods if we have tracking info.
  if (m_TrackingTransformNode.IsNotNull())
  {
    // Don't change the order of these.
    // Malti requires an initial hand-eye, so we use Tsai.
    m_HandEyeMatrices[0][TSAI_1989] = DoTsaiHandEye(0, false);
    m_HandEyeMatrices[0][KANG_2014] = DoKangHandEye(0, false);
    m_HandEyeMatrices[0][MALTI_2013] = DoMaltiHandEye(0, false);
    if (m_ImageNode[1].IsNull())
    {
      m_HandEyeMatrices[0][NON_LINEAR_EXTRINSIC] = DoFullExtrinsicHandEye(0, false);
    }

    if (m_ImageNode[1].IsNotNull())
    {
      // Don't change the order of these.
      // Malti requires an initial hand-eye, so we use Tsai.
      m_HandEyeMatrices[1][TSAI_1989] = DoTsaiHandEye(1, false);
      m_HandEyeMatrices[1][KANG_2014] = DoKangHandEye(1, false);
      m_HandEyeMatrices[1][MALTI_2013] = DoMaltiHandEye(1, false);

      DoFullExtrinsicHandEyeInStereo(m_HandEyeMatrices[0][NON_LINEAR_EXTRINSIC],
                                     m_HandEyeMatrices[1][NON_LINEAR_EXTRINSIC],
                                     false
                                    );
    } // end if we are in stereo.

    if (m_ReferenceTrackingTransformNode.IsNotNull())
    {
      // Don't change the order of these.
      // Malti requires an initial hand-eye, so we use Tsai.
      m_ReferenceHandEyeMatrices[0][TSAI_1989] = DoTsaiHandEye(0, true);
      m_ReferenceHandEyeMatrices[0][KANG_2014] = DoKangHandEye(0, true);
      m_ReferenceHandEyeMatrices[0][MALTI_2013] = DoMaltiHandEye(0, true);
      if (m_ImageNode[1].IsNull())
      {
        m_ReferenceHandEyeMatrices[0][NON_LINEAR_EXTRINSIC] = DoFullExtrinsicHandEye(0, true);
      }

      if (m_ImageNode[1].IsNotNull())
      {
        // Don't change the order of these.
        // Malti requires an initial hand-eye, so we use Tsai.
        m_ReferenceHandEyeMatrices[1][TSAI_1989] = DoTsaiHandEye(1, true);
        m_ReferenceHandEyeMatrices[1][KANG_2014] = DoKangHandEye(1, true);
        m_ReferenceHandEyeMatrices[1][MALTI_2013] = DoMaltiHandEye(1, true);

        DoFullExtrinsicHandEyeInStereo(m_ReferenceHandEyeMatrices[0][NON_LINEAR_EXTRINSIC],
                                       m_ReferenceHandEyeMatrices[1][NON_LINEAR_EXTRINSIC],
                                       true
                                      );

      } // end if we are in stereo
    } // end if we have a reference matrix.
  } // end if we have tracking data.

  MITK_INFO << "Calibrating - DONE.";
  return rms;
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::Save()
{
  if (m_ImageNode[0].IsNull())
  {
    mitkThrow() << "Left image should never be NULL.";
  }

  if (m_OutputDirName.empty())
  {
    mitkThrow() << "Empty output directory name.";
  }

  MITK_INFO << "Saving calibration to:" << m_OutputDirName << ":";

  niftk::SaveNifTKIntrinsics(m_Intrinsic[0], m_Distortion[0], m_OutputDirName + "calib.left.intrinsics.txt");
  this->SaveImages("calib.left.images.", m_OriginalImages[0]);
  this->SavePoints("calib.left.points.", m_Points[0]);

  if (m_ImageNode[1].IsNotNull())
  {
    niftk::SaveNifTKIntrinsics(
      m_Intrinsic[1], m_Distortion[1], m_OutputDirName + "calib.right.intrinsics.txt");

    niftk::SaveNifTKStereoExtrinsics(
      m_RightToLeftRotation, m_RightToLeftTranslation, m_OutputDirName + "calib.r2l.txt");

    this->SaveImages("calib.right.images.", m_OriginalImages[1]);
    this->SavePoints("calib.right.points.", m_Points[1]);
  }

  if (m_TrackingTransformNode.IsNotNull())
  {
    int counter = 0;
    std::list<cv::Matx44d >::const_iterator iter;
    for (iter = m_TrackingMatrices.begin();
         iter != m_TrackingMatrices.end();
         ++iter
         )
    {
      std::ostringstream fileName;
      fileName << m_OutputDirName << "calib.tracking." << counter++ << ".4x4";
      niftk::Save4x4Matrix(*iter, fileName.str());
    }

    // We deliberately output all hand-eye matrices, and additionally, whichever one was preferred method.
    niftk::Save4x4Matrix(m_HandEyeMatrices[0][0], m_OutputDirName + "calib.left.handeye.tsai.txt");
    niftk::Save4x4Matrix(m_HandEyeMatrices[0][1], m_OutputDirName + "calib.left.handeye.kang.txt");
    niftk::Save4x4Matrix(m_HandEyeMatrices[0][2], m_OutputDirName + "calib.left.handeye.malti.txt");
    niftk::Save4x4Matrix(m_HandEyeMatrices[0][3], m_OutputDirName + "calib.left.handeye.allextrinsic.txt");
    niftk::Save4x4Matrix(m_HandEyeMatrices[0][m_HandeyeMethod], m_OutputDirName + "calib.left.handeye.txt");

    niftk::SaveRigidParams(m_HandEyeMatrices[0][0], m_OutputDirName + "calib.left.handeye.tsai.params.txt");
    niftk::SaveRigidParams(m_HandEyeMatrices[0][1], m_OutputDirName + "calib.left.handeye.kang.params.txt");
    niftk::SaveRigidParams(m_HandEyeMatrices[0][2], m_OutputDirName + "calib.left.handeye.malti.params.txt");
    niftk::SaveRigidParams(m_HandEyeMatrices[0][3], m_OutputDirName + "calib.left.handeye.allextrinsic.params.txt");
    niftk::SaveRigidParams(m_HandEyeMatrices[0][m_HandeyeMethod], m_OutputDirName + "calib.left.handeye.params.txt");

    if (m_ImageNode[1].IsNotNull())
    {
      // We deliberately output all hand-eye matrices, and additionally, whichever one was preferred method.
      niftk::Save4x4Matrix(m_HandEyeMatrices[1][0], m_OutputDirName
          + "calib.right.handeye.tsai.txt");
      niftk::Save4x4Matrix(m_HandEyeMatrices[1][1], m_OutputDirName
          + "calib.right.handeye.kang.txt");
      niftk::Save4x4Matrix(m_HandEyeMatrices[1][2], m_OutputDirName
          + "calib.right.handeye.malti.txt");
      niftk::Save4x4Matrix(m_HandEyeMatrices[1][3], m_OutputDirName
          + "calib.right.handeye.allextrinsic.txt");
      niftk::Save4x4Matrix(m_HandEyeMatrices[1][m_HandeyeMethod], m_OutputDirName
          + "calib.right.handeye.txt");

      niftk::SaveRigidParams(m_HandEyeMatrices[1][0], m_OutputDirName
          + "calib.right.handeye.tsai.params.txt");
      niftk::SaveRigidParams(m_HandEyeMatrices[1][1], m_OutputDirName
          + "calib.right.handeye.kang.params.txt");
      niftk::SaveRigidParams(m_HandEyeMatrices[1][2], m_OutputDirName
          + "calib.right.handeye.malti.params.txt");
      niftk::SaveRigidParams(m_HandEyeMatrices[1][3], m_OutputDirName
          + "calib.right.handeye.allextrinsic.params.txt");
      niftk::SaveRigidParams(m_HandEyeMatrices[1][m_HandeyeMethod], m_OutputDirName
          + "calib.right.handeye.params.txt");
    }

    if (m_ReferenceTrackingTransformNode.IsNotNull())
    {
      int counter = 0;
      std::list<cv::Matx44d >::const_iterator iter;
      for (iter = m_ReferenceTrackingMatrices.begin();
           iter != m_ReferenceTrackingMatrices.end();
           ++iter
           )
      {
        std::ostringstream fileName;
        fileName << m_OutputDirName << "calib.tracking.reference." << counter++ << ".4x4";
        niftk::Save4x4Matrix(*iter, fileName.str());
      }

      // We deliberately output all hand-eye matrices, and additionally, whichever one was preferred method.
      niftk::Save4x4Matrix(m_ReferenceHandEyeMatrices[0][0], m_OutputDirName
          + "calib.left.handeye.reference.tsai.txt");
      niftk::Save4x4Matrix(m_ReferenceHandEyeMatrices[0][1], m_OutputDirName
          + "calib.left.handeye.reference.kang.txt");
      niftk::Save4x4Matrix(m_ReferenceHandEyeMatrices[0][2], m_OutputDirName
          + "calib.left.handeye.reference.malti.txt");
      niftk::Save4x4Matrix(m_ReferenceHandEyeMatrices[0][3], m_OutputDirName
          + "calib.left.handeye.reference.allextrinsic.txt");
      niftk::Save4x4Matrix(m_ReferenceHandEyeMatrices[0][m_HandeyeMethod], m_OutputDirName
          + "calib.left.handeye.reference.txt");

      niftk::SaveRigidParams(m_ReferenceHandEyeMatrices[0][0], m_OutputDirName
          + "calib.left.handeye.reference.tsai.params.txt");
      niftk::SaveRigidParams(m_ReferenceHandEyeMatrices[0][1], m_OutputDirName
          + "calib.left.handeye.reference.kang.params.txt");
      niftk::SaveRigidParams(m_ReferenceHandEyeMatrices[0][2], m_OutputDirName
          + "calib.left.handeye.reference.malti.params.txt");
      niftk::SaveRigidParams(m_ReferenceHandEyeMatrices[0][3], m_OutputDirName
          + "calib.left.handeye.reference.allextrinsic.params.txt");
      niftk::SaveRigidParams(m_ReferenceHandEyeMatrices[0][m_HandeyeMethod], m_OutputDirName
          + "calib.left.handeye.reference.params.txt");

      if (m_ImageNode[1].IsNotNull())
      {
        // We deliberately output all hand-eye matrices, and additionally, whichever one was preferred method.
        niftk::Save4x4Matrix(m_ReferenceHandEyeMatrices[1][0], m_OutputDirName
            + "calib.right.handeye.reference.tsai.txt");
        niftk::Save4x4Matrix(m_ReferenceHandEyeMatrices[1][1], m_OutputDirName
            + "calib.right.handeye.reference.kang.txt");
        niftk::Save4x4Matrix(m_ReferenceHandEyeMatrices[1][2], m_OutputDirName
            + "calib.right.handeye.reference.malti.txt");
        niftk::Save4x4Matrix(m_ReferenceHandEyeMatrices[1][3], m_OutputDirName
            + "calib.right.handeye.reference.allextrinsic.txt");
        niftk::Save4x4Matrix(m_ReferenceHandEyeMatrices[1][m_HandeyeMethod], m_OutputDirName
            + "calib.right.handeye.reference.txt");

        niftk::SaveRigidParams(m_ReferenceHandEyeMatrices[1][0], m_OutputDirName
            + "calib.right.handeye.reference.tsai.params.txt");
        niftk::SaveRigidParams(m_ReferenceHandEyeMatrices[1][1], m_OutputDirName
            + "calib.right.handeye.reference.kang.params.txt");
        niftk::SaveRigidParams(m_ReferenceHandEyeMatrices[1][2], m_OutputDirName
            + "calib.right.handeye.reference.malti.params.txt");
        niftk::SaveRigidParams(m_ReferenceHandEyeMatrices[1][3], m_OutputDirName
            + "calib.right.handeye.reference.allextrinsic.params.txt");
        niftk::SaveRigidParams(m_ReferenceHandEyeMatrices[1][m_HandeyeMethod], m_OutputDirName
            + "calib.right.handeye.reference.params.txt");
      }
    } // end if we have a reference transform
  } // end if we have tracking info

  MITK_INFO << "Saving calibration to:" << m_OutputDirName << ": - DONE.";
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::SaveImages(const std::string& prefix,
  const std::list<
    std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat> >& images)
{
  int counter = 0;
  std::list<std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat> >::const_iterator iter;
  for (iter = images.begin();
       iter != images.end();
       ++iter
       )
  {
    std::ostringstream fileName;
    fileName << m_OutputDirName << prefix << counter++ << ".png";
    cv::imwrite(fileName.str(), (*iter).second);
  }
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::SavePoints(const std::string& prefix, const std::list<niftk::PointSet>& points)
{
  int counter = 0;
  std::list<niftk::PointSet>::const_iterator iter;
  for (iter = points.begin();
       iter != points.end();
       ++iter
       )
  {
    std::ostringstream fileName;
    fileName << m_OutputDirName << prefix << counter++ << ".txt";
    niftk::SavePointSet(*iter, fileName.str());
  }
}

} // end namespace

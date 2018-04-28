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
#include <mitkCameraIntrinsics.h>
#include <mitkCameraIntrinsicsProperty.h>
#include <mitkProperties.h>

#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>
#include <highgui.h>
#include <sstream>

// NifTK
#include <mitkOpenCVMaths.h>
#include <mitkOpenCVFileIOUtils.h>
#include <niftkFileHelper.h>
#include <niftkOpenCVImageConversion.h>
#include <niftkCoordinateAxesData.h>
#include <niftkUndistortion.h>
#include <niftkMatrixUtilities.h>
#include <niftkSystemTimeServiceRAII.h>
#include <QmitkIGIUtils.h>

// NiftyCal
#include <niftkNiftyCalTypes.h>
#include <niftkChessboardPointDetector.h>
#include <niftkCirclesPointDetector.h>
#include <niftkAprilTagsPointDetector.h>
#include <niftkTemplateCirclesPointDetector.h>
#include <niftkTemplateRingsPointDetector.h>
#include <niftkIOUtilities.h>
#include <niftkPointUtilities.h>
#include <niftkZhangCameraCalibration.h>
#include <niftkTsaiCameraCalibration.h>
#include <niftkStereoCameraCalibration.h>
#include <niftkIterativeMonoCameraCalibration.h>
#include <niftkIterativeStereoCameraCalibration.h>
#include <niftkHandEyeCalibration.h>
#include <niftkSideBySideDetector.h>

namespace niftk
{

const bool                NiftyCalVideoCalibrationManager::DefaultDoIterative(false);
const bool                NiftyCalVideoCalibrationManager::DefaultDo3DOptimisation(false);
const bool                NiftyCalVideoCalibrationManager::DefaultDoClustering(true);
const unsigned int        NiftyCalVideoCalibrationManager::DefaultNumberOfSnapshotsForCalibrating(10);
const double              NiftyCalVideoCalibrationManager::DefaultScaleFactorX(1);
const double              NiftyCalVideoCalibrationManager::DefaultScaleFactorY(1);
const unsigned int        NiftyCalVideoCalibrationManager::DefaultGridSizeX(14);
const unsigned int        NiftyCalVideoCalibrationManager::DefaultGridSizeY(10);
const std::string         NiftyCalVideoCalibrationManager::DefaultTagFamily("25h7");
const bool                NiftyCalVideoCalibrationManager::DefaultUpdateNodes(true);
const unsigned int        NiftyCalVideoCalibrationManager::DefaultMinimumNumberOfPoints(70);

const NiftyCalVideoCalibrationManager::CalibrationPatterns
  NiftyCalVideoCalibrationManager::DefaultCalibrationPattern(NiftyCalVideoCalibrationManager::CHESS_BOARD);

const NiftyCalVideoCalibrationManager::HandEyeMethod
  NiftyCalVideoCalibrationManager::DefaultHandEyeMethod(NiftyCalVideoCalibrationManager::TSAI_1989);

//-----------------------------------------------------------------------------
NiftyCalVideoCalibrationManager::NiftyCalVideoCalibrationManager()
: m_DataStorage(nullptr)
, m_TrackingTransformNode(nullptr)
, m_ModelTransformNode(nullptr)
, m_DoIterative(NiftyCalVideoCalibrationManager::DefaultDoIterative)
, m_Do3DOptimisation(NiftyCalVideoCalibrationManager::DefaultDo3DOptimisation)
, m_NumberOfSnapshotsForCalibrating(NiftyCalVideoCalibrationManager::DefaultNumberOfSnapshotsForCalibrating)
, m_ScaleFactorX(NiftyCalVideoCalibrationManager::DefaultScaleFactorX)
, m_ScaleFactorY(NiftyCalVideoCalibrationManager::DefaultScaleFactorY)
, m_GridSizeX(NiftyCalVideoCalibrationManager::DefaultGridSizeX)
, m_GridSizeY(NiftyCalVideoCalibrationManager::DefaultGridSizeY)
, m_CalibrationPattern(NiftyCalVideoCalibrationManager::DefaultCalibrationPattern)
, m_HandeyeMethod(NiftyCalVideoCalibrationManager::DefaultHandEyeMethod)
, m_TagFamily(NiftyCalVideoCalibrationManager::DefaultTagFamily)
, m_UpdateNodes(NiftyCalVideoCalibrationManager::DefaultUpdateNodes)
, m_ModelTransformFileName("")
, m_MinimumNumberOfPoints(NiftyCalVideoCalibrationManager::DefaultMinimumNumberOfPoints)
, m_CalibrationDirName("")
{
  m_ImageNode[0] = nullptr;
  m_ImageNode[1] = nullptr;

  // 4 different methods - see HandEyeMethod enum
  for (int i = 0; i < 4; i++)
  {
    m_HandEyeMatrices[0].push_back(cv::Matx44d::eye()); // left
    m_HandEyeMatrices[1].push_back(cv::Matx44d::eye()); // right
  }

  m_ModelToWorld = cv::Matx44d::eye();   // computed on the fly after each calibration
  m_StaticModelTransform = cv::Matx44d::eye(); // loaded in, say from a point based registration from chessboard to tracker space.

  m_ModelPointsToVisualise = mitk::PointSet::New();
  m_ModelPointsToVisualiseDataNode = mitk::DataNode::New();
  m_ModelPointsToVisualiseDataNode->SetData(m_ModelPointsToVisualise);
  m_ModelPointsToVisualiseDataNode->SetName("CalibrationModelPoints");
  m_ModelPointsToVisualiseDataNode->SetBoolProperty("helper object", true);

  // These must be initialised for the use-case where we load them from disk.
  m_Intrinsic[0] = cvCreateMat(3, 3, CV_64FC1);
  m_Intrinsic[0] = cv::Mat::eye(3, 3, CV_64FC1);
  m_Intrinsic[1] = cvCreateMat(3, 3, CV_64FC1);
  m_Intrinsic[1] = cv::Mat::eye(3, 3, CV_64FC1);
  m_Distortion[0] = cvCreateMat(1, 5, CV_64FC1);    // Note: NiftyCal and OpenCV by default use 5 params.
  m_Distortion[0] = cv::Mat::zeros(1, 5, CV_64FC1); // But MITK only uses 4, so we save/load 4 :-(
  m_Distortion[1] = cvCreateMat(1, 5, CV_64FC1);    // Note: NiftyCal and OpenCV by default use 5 params.
  m_Distortion[1] = cv::Mat::zeros(1, 5, CV_64FC1); // But MITK only uses 4, so we save/load 4 :-(
  m_LeftToRightRotationMatrix = cvCreateMat(3,3,CV_64FC1);
  m_LeftToRightRotationMatrix = cv::Mat::eye(3, 3, CV_64FC1);
  m_LeftToRightTranslationVector = cvCreateMat(3,1,CV_64FC1);
  m_LeftToRightTranslationVector = cv::Mat::eye(3, 1, CV_64FC1);
}


//-----------------------------------------------------------------------------
NiftyCalVideoCalibrationManager::~NiftyCalVideoCalibrationManager()
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->Remove(m_ModelPointsToVisualiseDataNode);

    for (int i = 0; i < m_TrackingMatricesDataNodes.size(); i++)
    {
      m_DataStorage->Remove(m_TrackingMatricesDataNodes[i]);
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
  if (!m_CalibrationDirName.empty())
  {
    this->UpdateDisplayNodes();
  }
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
  if (!m_CalibrationDirName.empty())
  {
    this->UpdateDisplayNodes();
  }
  this->Modified();
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer NiftyCalVideoCalibrationManager::GetRightImageNode() const
{
  return m_ImageNode[1];
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::SetTrackingTransformNode(mitk::DataNode::Pointer node)
{
  this->m_TrackingTransformNode = node;
  if (!m_CalibrationDirName.empty())
  {
    this->UpdateDisplayNodes();
  }
  this->UpdateCameraToWorldPosition();
  this->UpdateVisualisedPoints();
  this->Modified();
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::UpdateVisualisedPoints()
{
  this->UpdateVisualisedPoints(m_ModelToWorld);
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::UpdateVisualisedPoints(cv::Matx44d& transform)
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->Remove(m_ModelPointsToVisualiseDataNode);

    m_ModelPointsToVisualise->Clear();

    niftk::Model3D::const_iterator iter;
    for (iter = m_ModelPoints.begin();
         iter != m_ModelPoints.end();
         ++iter
         )
    {
      cv::Point3d p1 = (*iter).second.point;
      cv::Matx41d p2;
      p2(0, 0) = p1.x;
      p2(1, 0) = p1.y;
      p2(2, 0) = p1.z;
      p2(3, 0) = 1;
      cv::Matx41d p3 = transform * p2;

      mitk::Point3D p4;
      p4[0] = p3(0, 0);
      p4[1] = p3(1, 0);
      p4[2] = p3(2, 0);

      m_ModelPointsToVisualise->InsertPoint((*iter).first, p4);
    }

    m_DataStorage->Add(m_ModelPointsToVisualiseDataNode);
  }
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

  cv::Matx44d id = cv::Matx44d::eye();
  this->UpdateVisualisedPoints(id);
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
void NiftyCalVideoCalibrationManager::SetTemplateImageFileName(const std::string& fileName)
{
  if (!fileName.empty())
  {
    cv::Mat templateImage = cv::imread(fileName);
    if (templateImage.rows == 0 || templateImage.cols == 0)
    {
      mitkThrow() << "Failed to read template image:" << fileName;
    }

    cv::Mat templateImageGreyScale;
    cv::cvtColor(templateImage, templateImageGreyScale, CV_BGR2GRAY);

    m_TemplateImageFileName = fileName;
    m_TemplateImage = templateImageGreyScale;
    this->Modified();
  }
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::SetModelTransformFileName(
    const std::string& fileName)
{
  if (!fileName.empty())
  {
    m_StaticModelTransform = niftk::LoadMatrix(fileName);
    m_ModelTransformFileName = fileName;
    this->Modified();
  }
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::SetOutputPrefixName(const std::string& dirName)
{
  if (dirName.empty())
  {
    mitkThrow() << "Empty output directory name.";
  }

  m_OutputPrefixName = (dirName + niftk::GetFileSeparator());
  this->Modified();
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::UpdateCameraToWorldPosition()
{
  if (m_TrackingTransformNode.IsNotNull())
  {
    CoordinateAxesData::Pointer tracking = dynamic_cast<CoordinateAxesData*>(
          m_TrackingTransformNode->GetData());
    if (tracking.IsNull())
    {
      mitkThrow() << "Tracking node contains null tracking matrix.";
    }
    vtkSmartPointer<vtkMatrix4x4> trackingMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
    tracking->GetVtkMatrix(*trackingMatrix);

    vtkSmartPointer<vtkMatrix4x4> handEyeMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
    for (int r = 0; r < 4; r++)
    {
      for (int c = 0; c < 4; c++)
      {
        handEyeMatrix->SetElement(r, c, m_HandEyeMatrices[0][m_HandeyeMethod](r, c));
      }
    }

    handEyeMatrix->Invert();

    vtkSmartPointer<vtkMatrix4x4> cameraToWorldMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
    vtkMatrix4x4::Multiply4x4(trackingMatrix, handEyeMatrix, cameraToWorldMatrix);

    std::string cameraToWorldNode = "niftk.ls.cameratoworld";
    mitk::DataNode::Pointer node = m_DataStorage->GetNamedNode(cameraToWorldNode);
    if (node.IsNull())
    {
      node = mitk::DataNode::New();
      node->SetName(cameraToWorldNode);
      node->SetBoolProperty("helper object", true);
    }

    CoordinateAxesData::Pointer coords = dynamic_cast<CoordinateAxesData*>(node->GetData());
    if (coords.IsNull())
    {
      coords = CoordinateAxesData::New();
      node->SetData(coords);
      m_DataStorage->Add(node);
    }

    coords->SetVtkMatrix(*cameraToWorldMatrix);
    node->Modified();
  }
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
  for (int i = 0; i < m_TrackingMatricesDataNodes.size(); i++)
  {
    if (m_DataStorage.IsNotNull()
        && m_DataStorage->Exists(m_TrackingMatricesDataNodes[i])
       )
    {
      m_DataStorage->Remove(m_TrackingMatricesDataNodes[i]);
    }
  }
  m_TrackingMatricesDataNodes.clear();
  m_TrackingMatrices.clear();
  m_ModelTrackingMatrices.clear();

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
std::list<cv::Matx44d> NiftyCalVideoCalibrationManager::ExtractTrackingMatrices()
{
  std::list<cv::Matx44d> trackingMatrices;

  std::list<cv::Matx44d>::const_iterator trackingIter;
  for (trackingIter = m_TrackingMatrices.begin();
       trackingIter != m_TrackingMatrices.end();
       ++trackingIter
       )
  {
    trackingMatrices.push_back(*trackingIter);
  }

  return trackingMatrices;
}


//-----------------------------------------------------------------------------
std::list<cv::Matx44d> NiftyCalVideoCalibrationManager::ExtractModelMatrices()
{
  std::list<cv::Matx44d> modelMatrices;

  std::list<cv::Matx44d>::const_iterator modelIter;
  for (modelIter = m_ModelTrackingMatrices.begin();
       modelIter != m_ModelTrackingMatrices.end();
       ++modelIter
       )
  {
    modelMatrices.push_back(*modelIter);
  }

  return modelMatrices;
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
cv::Matx44d NiftyCalVideoCalibrationManager::DoTsaiHandEye(int imageIndex)
{
  cv::Matx21d residual;
  residual(0, 0) = 0;
  residual(1, 0) = 0;

  std::list<cv::Matx44d> cameraMatrices = this->ExtractCameraMatrices(imageIndex);
  std::list<cv::Matx44d> trackingMatrices = this->ExtractTrackingMatrices();

  // This method needs at least 3 camera and tracking matrices,
  // corresponding to 2 movements, rotating about 2 independent axes.
  // eg. start, move by Rx, move by Ry = 3 posns.

  // And this method checks the number of tracking and hand matrices.
  cv::Matx44d handEye =
    niftk::CalculateHandEyeUsingTsaisMethod(
      trackingMatrices,
      cameraMatrices,
      residual
      );

  return handEye;
}


//-----------------------------------------------------------------------------
cv::Matx44d NiftyCalVideoCalibrationManager::DoShahidiHandEye(int imageIndex)
{
  std::list<cv::Matx44d> cameraMatrices = this->ExtractCameraMatrices(imageIndex);
  std::list<cv::Matx44d> trackingMatrices = this->ExtractTrackingMatrices();
  std::list<cv::Matx44d> modelTrackingMatrices = this->ExtractModelMatrices();

  if (cameraMatrices.size() != trackingMatrices.size())
  {
    mitkThrow() << "Number of camera matrices:" << cameraMatrices.size()
      << ", is not equal to the number of tracking matrices:" << trackingMatrices.size();
  }
  if (cameraMatrices.size() != modelTrackingMatrices.size())
  {
    mitkThrow() << "Number of camera matrices:" << cameraMatrices.size()
      << ", is not equal to the number of model (chessboard) tracking matrices:" << modelTrackingMatrices.size();
  }

  std::list<cv::Matx44d> handEyeMatrices;

  std::list<cv::Matx44d>::const_iterator cIter;
  std::list<cv::Matx44d>::const_iterator tIter;
  std::list<cv::Matx44d>::const_iterator mIter;

  for (cIter = cameraMatrices.begin(),
       tIter = trackingMatrices.begin(),
       mIter = modelTrackingMatrices.begin();
       cIter != cameraMatrices.end()
       && tIter != trackingMatrices.end()
       && mIter != modelTrackingMatrices.end();
       cIter++, tIter++, mIter++
       )
  {
    std::list<cv::Matx44d> cTmp;
    cTmp.push_back(*cIter);

    std::list<cv::Matx44d> tTmp;
    tTmp.push_back(*tIter);

    // i.e. calculate handEye, with 1 set of matrices at a time.
    cv::Matx44d handEye = niftk::CalculateHandEyeByDirectMatrixMultiplication(
      (*mIter * m_StaticModelTransform), // chessboard tracking matrix * static chessboard transform
      tTmp,
      cTmp
      );

    handEyeMatrices.push_back(handEye);
  }

  cv::Matx44d averaged = niftk::AverageMatricesUsingEigenValues(handEyeMatrices);

  return averaged;
}


//-----------------------------------------------------------------------------
cv::Matx44d NiftyCalVideoCalibrationManager::GetInitialHandEye(int imageIndex)
{
  cv::Matx44d handEye;
  if (m_TrackingMatrices.size() > 1)
  {
    handEye = m_HandEyeMatrices[imageIndex][TSAI_1989];
  }
  else
  {
    handEye = m_HandEyeMatrices[imageIndex][SHAHIDI_2002];
  }
  return handEye;
}


//-----------------------------------------------------------------------------
cv::Matx44d NiftyCalVideoCalibrationManager::GetInitialModelToWorld()
{
  cv::Matx44d modelToWorld;
  if (m_TrackingMatrices.size() > 1)
  {
    modelToWorld = this->GetModelToWorld(m_HandEyeMatrices[0][TSAI_1989]);
  }
  else
  {
    modelToWorld = this->GetModelToWorld(m_HandEyeMatrices[0][SHAHIDI_2002]);
  }
  return modelToWorld;
}


//-----------------------------------------------------------------------------
cv::Matx44d NiftyCalVideoCalibrationManager::GetModelToWorld(const cv::Matx44d& handEye)
{
  std::list<cv::Matx44d> cameraMatrices = this->ExtractCameraMatrices(0);
  if (cameraMatrices.empty())
  {
    mitkThrow() << "Empty list of camera matrices.";
  }

  std::list<cv::Matx44d> trackingMatrices = this->ExtractTrackingMatrices();
  if (trackingMatrices.empty())
  {
    mitkThrow() << "Empty list of tracking matrices.";
  }

  cv::Matx44d modelToWorld = niftk::CalculateAverageModelToWorld(
        handEye,
        trackingMatrices,
        cameraMatrices
        );

  return modelToWorld;
}


//-----------------------------------------------------------------------------
cv::Matx44d NiftyCalVideoCalibrationManager::DoMaltiHandEye(int imageIndex)
{
  double reprojectionRMS = 0;

  std::list<cv::Matx44d> trackingMatrices = this->ExtractTrackingMatrices();

  // We clone them, so we dont modify the member variables m_Intrinsic, m_Distortion.
  cv::Mat intrinsic = m_Intrinsic[imageIndex].clone();
  cv::Mat distortion = m_Distortion[imageIndex].clone();

  cv::Matx44d handEye = this->GetInitialHandEye(imageIndex);
  cv::Matx44d modelToWorld = this->GetInitialModelToWorld();

  niftk::CalculateHandEyeUsingMaltisMethod(m_ModelPoints,
                                           m_Points[imageIndex],
                                           trackingMatrices,
                                           intrinsic,
                                           distortion,
                                           handEye,
                                           modelToWorld,
                                           reprojectionRMS
                                          );

  std::ostringstream message;
  message << "Malti mono[" << imageIndex << "]:" << reprojectionRMS << " pixels" << std::endl;
  m_CalibrationResult += message.str();

  return handEye;
}


//-----------------------------------------------------------------------------
cv::Matx44d NiftyCalVideoCalibrationManager::DoFullExtrinsicHandEye(int imageIndex)
{
  double reprojectionRMS = 0;

  std::list<cv::Matx44d> trackingMatrices = this->ExtractTrackingMatrices();

  cv::Matx44d handEye = this->GetInitialHandEye(imageIndex);
  cv::Matx44d modelToWorld = this->GetInitialModelToWorld();

  niftk::CalculateHandEyeByOptimisingAllExtrinsic(m_ModelPoints,
                                                  m_Points[imageIndex],
                                                  trackingMatrices,
                                                  m_Intrinsic[imageIndex],
                                                  m_Distortion[imageIndex],
                                                  handEye,
                                                  modelToWorld,
                                                  reprojectionRMS
                                                 );

  std::ostringstream message;
  message << "Non-Linear Ext mono[" << imageIndex << "]:" << reprojectionRMS << " pixels" << std::endl;
  m_CalibrationResult += message.str();

  return handEye;
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::DoFullExtrinsicHandEyeInStereo(cv::Matx44d& leftHandEye,
                                                                     cv::Matx44d& rightHandEye
                                                                     )
{
  double reprojectionRMS = 0;

  std::list<cv::Matx44d> trackingMatrices = this->ExtractTrackingMatrices();

  cv::Matx44d stereoExtrinsics = niftk::RotationAndTranslationToMatrix(
        m_LeftToRightRotationMatrix, m_LeftToRightTranslationVector);

  cv::Matx44d handEye = this->GetInitialHandEye(0);
  cv::Matx44d modelToWorld = this->GetInitialModelToWorld();

  niftk::CalculateHandEyeInStereoByOptimisingAllExtrinsic(m_ModelPoints,
                                                          m_Points[0],
                                                          m_Intrinsic[0],
                                                          m_Distortion[0],
                                                          m_Points[1],
                                                          m_Intrinsic[1],
                                                          m_Distortion[1],
                                                          trackingMatrices,
                                                          m_Do3DOptimisation,
                                                          handEye,
                                                          modelToWorld,
                                                          stereoExtrinsics,
                                                          reprojectionRMS
                                                         );
  leftHandEye = handEye;
  rightHandEye = (stereoExtrinsics.inv()) * handEye;

  std::ostringstream message;
  message << "Non-Linear Ext stereo: " << reprojectionRMS << " pixels" << std::endl;
  m_CalibrationResult += message.str();
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
  if (image.channels() == 1)
  {
    image.copyTo(outputImage);
  }
  else if (image.channels() == 3)
  {
    cv::cvtColor(image, outputImage, CV_RGB2GRAY);
  }
  else if (image.channels() == 4)
  {
    cv::cvtColor(image, outputImage, CV_RGBA2GRAY);
  }
  else
  {
    mitkThrow() << "Input image should be 1 (grey scale), 3 (RGB) or 4 (RGBA) channel.";
  }

  m_ImageSize.width = image.cols;
  m_ImageSize.height = image.rows;
}


//-----------------------------------------------------------------------------
bool NiftyCalVideoCalibrationManager::ExtractPoints(int imageIndex, const cv::Mat& image)
{
  bool isSuccessful = false;

  cv::Point2d noScaleFactors(1, 1);

  cv::Point2d scaleFactors;
  scaleFactors.x = m_ScaleFactorX;
  scaleFactors.y = m_ScaleFactorY;

  bool doRescaleAfterPointExtraction = true;
  int offsetForTemplateMatching = 10;

  cv::Mat copyOfImage1 = image.clone(); // Remember OpenCV reference counting.
  cv::Mat copyOfImage2 = image.clone(); // Remember OpenCV reference counting.

  int clusteringFlag = 0;
  if (m_DoClustering)
  {
    clusteringFlag = cv::CALIB_CB_CLUSTERING;
  }

  // Watch out: OpenCV reference counts the image data block.
  // So, if you create two cv::Mat, using say the copy constructor
  // or assignment operator, both cv::Mat point to the same memory
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

    niftk::ChessboardPointDetector *chessboardDetector1 = new niftk::ChessboardPointDetector(internalCorners);
    chessboardDetector1->SetImageScaleFactor(scaleFactors, doRescaleAfterPointExtraction);
    chessboardDetector1->SetImage(&copyOfImage1);
    chessboardDetector1->SetCaching(true);

    niftk::PointSet points = chessboardDetector1->GetPoints();
    if (points.size() == m_GridSizeX * m_GridSizeY)
    {
      isSuccessful = true;
      m_Points[imageIndex].push_back(points);

      std::shared_ptr<niftk::IPoint2DDetector> originalDetector(chessboardDetector1);
      m_OriginalImages[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(originalDetector, copyOfImage1));
      dynamic_cast<niftk::ChessboardPointDetector*>(
        m_OriginalImages[imageIndex].back().first.get())->SetImage(&(m_OriginalImages[imageIndex].back().second));

      niftk::ChessboardPointDetector *chessboardDetector2 =
        new niftk::ChessboardPointDetector(internalCorners);
      chessboardDetector2->SetImageScaleFactor(scaleFactors, doRescaleAfterPointExtraction);
      chessboardDetector2->SetImage(&copyOfImage2);
      chessboardDetector2->SetCaching(false);

      std::shared_ptr<niftk::IPoint2DDetector> warpedDetector(chessboardDetector2);
      m_ImagesForWarping[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(warpedDetector, copyOfImage2));
      dynamic_cast<niftk::ChessboardPointDetector*>(
        m_ImagesForWarping[imageIndex].back().first.get())->SetImage(&(m_ImagesForWarping[imageIndex].back().second));
    }
  }
  else if (m_CalibrationPattern == CIRCLE_GRID)
  {
    cv::Size2i internalCorners(m_GridSizeX, m_GridSizeY);

    niftk::CirclesPointDetector *circlesDetector1 =
      new niftk::CirclesPointDetector(internalCorners, cv::CALIB_CB_ASYMMETRIC_GRID | clusteringFlag);
    circlesDetector1->SetImageScaleFactor(scaleFactors, doRescaleAfterPointExtraction);
    circlesDetector1->SetImage(&copyOfImage1);
    circlesDetector1->SetCaching(true);

    niftk::PointSet points = circlesDetector1->GetPoints();
    if (points.size() == m_GridSizeX * m_GridSizeY)
    {
      isSuccessful = true;
      m_Points[imageIndex].push_back(points);

      std::shared_ptr<niftk::IPoint2DDetector> originalDetector(circlesDetector1);
      m_OriginalImages[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(originalDetector, copyOfImage1));
      dynamic_cast<niftk::CirclesPointDetector*>(
        m_OriginalImages[imageIndex].back().first.get())->SetImage(&(m_OriginalImages[imageIndex].back().second));

      niftk::CirclesPointDetector *circlesDetector2 =
        new niftk::CirclesPointDetector(internalCorners, cv::CALIB_CB_ASYMMETRIC_GRID | clusteringFlag);
      circlesDetector2->SetImageScaleFactor(noScaleFactors);
      circlesDetector2->SetImage(&copyOfImage2);
      circlesDetector2->SetCaching(false);

      std::shared_ptr<niftk::IPoint2DDetector> warpedDetector(circlesDetector2);
      m_ImagesForWarping[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(warpedDetector, copyOfImage2));
      dynamic_cast<niftk::CirclesPointDetector*>(
        m_ImagesForWarping[imageIndex].back().first.get())->SetImage(&(m_ImagesForWarping[imageIndex].back().second));
    }
  }
  else if (m_CalibrationPattern == APRIL_TAGS)
  {
    niftk::AprilTagsPointDetector *aprilTagsDetector1 =
      new niftk::AprilTagsPointDetector(false, m_TagFamily, 0, 0.8);
    aprilTagsDetector1->SetImageScaleFactor(scaleFactors, doRescaleAfterPointExtraction);
    aprilTagsDetector1->SetImage(&copyOfImage1);
    aprilTagsDetector1->SetCaching(true);

    niftk::PointSet points = aprilTagsDetector1->GetPoints();
    if (points.size() >= m_MinimumNumberOfPoints)
    {
      isSuccessful = true;
      m_Points[imageIndex].push_back(points);

      std::shared_ptr<niftk::IPoint2DDetector> originalDetector(aprilTagsDetector1);
      m_OriginalImages[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(originalDetector, copyOfImage1));
      dynamic_cast<niftk::AprilTagsPointDetector*>(
        m_OriginalImages[imageIndex].back().first.get())->SetImage(&(m_OriginalImages[imageIndex].back().second));

      niftk::AprilTagsPointDetector *aprilTagsDetector2 =
        new niftk::AprilTagsPointDetector(false, m_TagFamily, 0, 0.8);
      aprilTagsDetector2->SetImageScaleFactor(scaleFactors, doRescaleAfterPointExtraction);
      aprilTagsDetector2->SetImage(&copyOfImage2);
      aprilTagsDetector2->SetCaching(false);

      std::shared_ptr<niftk::IPoint2DDetector> warpedDetector(aprilTagsDetector2);
      m_ImagesForWarping[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(warpedDetector, copyOfImage2));
      dynamic_cast<niftk::AprilTagsPointDetector*>(
        m_ImagesForWarping[imageIndex].back().first.get())->SetImage(&(m_ImagesForWarping[imageIndex].back().second));
    }
  }
  else if (m_CalibrationPattern == TEMPLATE_MATCHING_CIRCLES)
  {
    cv::Size2i internalCorners(m_GridSizeX, m_GridSizeY);
    cv::Size2i offsetIfNotIterative(m_TemplateImage.cols / offsetForTemplateMatching,
                                    m_TemplateImage.rows / offsetForTemplateMatching);
    unsigned long int maxArea = m_TemplateImage.cols * m_TemplateImage.rows;

    niftk::TemplateCirclesPointDetector *circlesTemplateDetector1
        = new niftk::TemplateCirclesPointDetector(internalCorners, offsetIfNotIterative,
                                                  cv::CALIB_CB_ASYMMETRIC_GRID
                                                  | clusteringFlag);
    circlesTemplateDetector1->SetImage(&copyOfImage1);
    circlesTemplateDetector1->SetImageScaleFactor(scaleFactors, doRescaleAfterPointExtraction);
    circlesTemplateDetector1->SetTemplateImage(&m_TemplateImage);
    circlesTemplateDetector1->SetReferenceImage(&m_ReferenceDataForIterativeCalib.first);
    circlesTemplateDetector1->SetReferencePoints(m_ReferenceDataForIterativeCalib.second);
    circlesTemplateDetector1->SetMaxAreaInPixels(maxArea);
    circlesTemplateDetector1->SetUseContours(true);
    circlesTemplateDetector1->SetUseInternalResampling(true);
    circlesTemplateDetector1->SetUseTemplateMatching(true);
    circlesTemplateDetector1->SetCaching(true);

    niftk::PointSet points = circlesTemplateDetector1->GetPoints();

    if (points.size() == m_GridSizeX * m_GridSizeY)
    {
      isSuccessful = true;
      m_Points[imageIndex].push_back(points);

      std::shared_ptr<niftk::IPoint2DDetector> originalDetector(circlesTemplateDetector1);
      m_OriginalImages[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(originalDetector, copyOfImage1));
      dynamic_cast<niftk::TemplateCirclesPointDetector*>(
        m_OriginalImages[imageIndex].back().first.get())->SetImage(&(m_OriginalImages[imageIndex].back().second));

      niftk::TemplateCirclesPointDetector *circlesTemplateDetector2
          = new niftk::TemplateCirclesPointDetector(internalCorners,
                                                    offsetIfNotIterative,
                                                    cv::CALIB_CB_ASYMMETRIC_GRID
                                                    | clusteringFlag
                                                    );

      circlesTemplateDetector2->SetImage(&copyOfImage2);
      circlesTemplateDetector2->SetImageScaleFactor(noScaleFactors);
      circlesTemplateDetector2->SetTemplateImage(&m_TemplateImage);
      circlesTemplateDetector2->SetReferenceImage(&m_ReferenceDataForIterativeCalib.first);
      circlesTemplateDetector2->SetReferencePoints(m_ReferenceDataForIterativeCalib.second);
      circlesTemplateDetector2->SetMaxAreaInPixels(maxArea);
      circlesTemplateDetector2->SetUseContours(false);
      circlesTemplateDetector2->SetUseInternalResampling(false);
      circlesTemplateDetector2->SetUseTemplateMatching(true);
      circlesTemplateDetector2->SetCaching(false);
      circlesTemplateDetector2->SetInitialGuess(points);

      std::shared_ptr<niftk::IPoint2DDetector> warpedDetector(circlesTemplateDetector2);
      m_ImagesForWarping[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(warpedDetector, copyOfImage2));
      dynamic_cast<niftk::TemplateCirclesPointDetector*>(
        m_ImagesForWarping[imageIndex].back().first.get())->SetImage(&(m_ImagesForWarping[imageIndex].back().second));
    }
  }
  else if (m_CalibrationPattern == TEMPLATE_MATCHING_RINGS)
  {
    cv::Size2i internalCorners(m_GridSizeX, m_GridSizeY);
    cv::Size2i offsetIfNotIterative(m_TemplateImage.cols / offsetForTemplateMatching,
                                    m_TemplateImage.rows / offsetForTemplateMatching);
    unsigned long int maxArea = m_TemplateImage.cols * m_TemplateImage.rows;

    niftk::TemplateRingsPointDetector *ringsTemplateDetector1
        = new niftk::TemplateRingsPointDetector(internalCorners,
                                                offsetIfNotIterative,
                                                cv::CALIB_CB_ASYMMETRIC_GRID
                                                | clusteringFlag
                                                );

    ringsTemplateDetector1->SetImage(&copyOfImage1);
    ringsTemplateDetector1->SetImageScaleFactor(scaleFactors, doRescaleAfterPointExtraction);
    ringsTemplateDetector1->SetTemplateImage(&m_TemplateImage);
    ringsTemplateDetector1->SetReferenceImage(&m_ReferenceDataForIterativeCalib.first);
    ringsTemplateDetector1->SetReferencePoints(m_ReferenceDataForIterativeCalib.second);
    ringsTemplateDetector1->SetMaxAreaInPixels(maxArea);
    ringsTemplateDetector1->SetUseContours(true);
    ringsTemplateDetector1->SetUseInternalResampling(true);
    ringsTemplateDetector1->SetUseTemplateMatching(true);
    ringsTemplateDetector1->SetCaching(true);

    niftk::PointSet points = ringsTemplateDetector1->GetPoints();

    if (points.size() == m_GridSizeX * m_GridSizeY)
    {
      isSuccessful = true;
      m_Points[imageIndex].push_back(points);

      std::shared_ptr<niftk::IPoint2DDetector> originalDetector(ringsTemplateDetector1);
      m_OriginalImages[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(originalDetector, copyOfImage1));
      dynamic_cast<niftk::TemplateRingsPointDetector*>(
        m_OriginalImages[imageIndex].back().first.get())->SetImage(&(m_OriginalImages[imageIndex].back().second));

      niftk::TemplateRingsPointDetector *ringsTemplateDetector2
          = new niftk::TemplateRingsPointDetector(internalCorners,
                                                  offsetIfNotIterative,
                                                  cv::CALIB_CB_ASYMMETRIC_GRID
                                                  | clusteringFlag
                                                 );

      ringsTemplateDetector2->SetImage(&copyOfImage2);
      ringsTemplateDetector2->SetImageScaleFactor(scaleFactors, doRescaleAfterPointExtraction);
      ringsTemplateDetector2->SetTemplateImage(&m_TemplateImage);
      ringsTemplateDetector2->SetReferenceImage(&m_ReferenceDataForIterativeCalib.first);
      ringsTemplateDetector2->SetReferencePoints(m_ReferenceDataForIterativeCalib.second);
      ringsTemplateDetector2->SetMaxAreaInPixels(maxArea);
      ringsTemplateDetector2->SetUseContours(false);
      ringsTemplateDetector2->SetUseInternalResampling(false);
      ringsTemplateDetector2->SetUseTemplateMatching(true);
      ringsTemplateDetector2->SetCaching(false);
      ringsTemplateDetector2->SetInitialGuess(points);

      std::shared_ptr<niftk::IPoint2DDetector> warpedDetector(ringsTemplateDetector2);
      m_ImagesForWarping[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(warpedDetector, copyOfImage2));
      dynamic_cast<niftk::TemplateRingsPointDetector*>(
        m_ImagesForWarping[imageIndex].back().first.get())->SetImage(&(m_ImagesForWarping[imageIndex].back().second));
    }
  }
  else if (m_CalibrationPattern == TEMPLATE_MATCHING_NON_COPLANAR_CIRCLES)
  {
    cv::Size2i internalCorners(m_GridSizeX, m_GridSizeY);
    cv::Size2i offsetIfNotIterative(m_TemplateImage.cols / offsetForTemplateMatching,
                                    m_TemplateImage.rows / offsetForTemplateMatching);
    unsigned long int maxArea = m_TemplateImage.cols * m_TemplateImage.rows;

    // We set no scaleFactors, as this method uses a two phase
    // decorator pattern for the Tsai's method.
    // So, the SideBySide detector uses the scale factors
    // to scale up the image, as appropriate, so that each
    // point detector already has a scaled image.
    // So, each point detector requires no scale factors.

    std::unique_ptr<niftk::TemplateCirclesPointDetector> l1(
      new niftk::TemplateCirclesPointDetector(internalCorners,
                                              offsetIfNotIterative,
                                              cv::CALIB_CB_ASYMMETRIC_GRID
                                              | clusteringFlag
                                              ));

    l1->SetImageScaleFactor(noScaleFactors);
    l1->SetTemplateImage(&m_TemplateImage);
    l1->SetReferenceImage(&m_ReferenceDataForIterativeCalib.first);
    l1->SetReferencePoints(m_ReferenceDataForIterativeCalib.second);
    l1->SetMaxAreaInPixels(maxArea);
    l1->SetUseContours(true);
    l1->SetUseInternalResampling(true);
    l1->SetUseTemplateMatching(true);
    l1->SetCaching(true);

    std::unique_ptr<niftk::TemplateCirclesPointDetector> r1(
      new niftk::TemplateCirclesPointDetector(internalCorners,
                                              offsetIfNotIterative,
                                              cv::CALIB_CB_ASYMMETRIC_GRID
                                              | clusteringFlag
                                              ));

    r1->SetImageScaleFactor(noScaleFactors);
    r1->SetTemplateImage(&m_TemplateImage);
    r1->SetReferenceImage(&m_ReferenceDataForIterativeCalib.first);
    r1->SetReferencePoints(m_ReferenceDataForIterativeCalib.second);
    r1->SetMaxAreaInPixels(maxArea);
    r1->SetUseContours(true);
    r1->SetUseInternalResampling(true);
    r1->SetUseTemplateMatching(true);
    r1->SetCaching(true);

    std::unique_ptr<niftk::PointDetector> l2(l1.release());
    std::unique_ptr<niftk::PointDetector> r2(r1.release());

    std::unique_ptr<niftk::SideBySideDetector> s1(new niftk::SideBySideDetector(l2, r2));
    s1->SetImage(&copyOfImage1);
    s1->SetImageScaleFactor(scaleFactors, doRescaleAfterPointExtraction);

    std::unique_ptr<niftk::PointDetector> s2(s1.release());

    niftk::PointSet points = s2->GetPoints();

    if (points.size() == 2 * m_GridSizeX * m_GridSizeY)
    {
      isSuccessful = true;
      m_Points[imageIndex].push_back(points);

      std::shared_ptr<niftk::IPoint2DDetector> originalDetector(s2.release());

      m_OriginalImages[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(originalDetector, copyOfImage1));

      dynamic_cast<niftk::SideBySideDetector*>(
        m_OriginalImages[imageIndex].back().first.get())->SetImage(&(m_OriginalImages[imageIndex].back().second));

      std::unique_ptr<niftk::TemplateCirclesPointDetector> l3(
        new niftk::TemplateCirclesPointDetector(internalCorners,
                                                offsetIfNotIterative,
                                                cv::CALIB_CB_ASYMMETRIC_GRID
                                                | clusteringFlag
                                                ));

      l3->SetImageScaleFactor(noScaleFactors);
      l3->SetTemplateImage(&m_TemplateImage);
      l3->SetReferenceImage(&m_ReferenceDataForIterativeCalib.first);
      l3->SetReferencePoints(m_ReferenceDataForIterativeCalib.second);
      l3->SetMaxAreaInPixels(maxArea);
      l3->SetUseContours(true);
      l3->SetUseInternalResampling(false);
      l3->SetUseTemplateMatching(false);
      l3->SetCaching(false);
      l3->SetInitialGuess(points);

      std::unique_ptr<niftk::TemplateCirclesPointDetector> r3(
        new niftk::TemplateCirclesPointDetector(internalCorners,
                                                offsetIfNotIterative,
                                                cv::CALIB_CB_ASYMMETRIC_GRID
                                                | clusteringFlag
                                                ));

      r3->SetImageScaleFactor(noScaleFactors);
      r3->SetTemplateImage(&m_TemplateImage);
      r3->SetReferenceImage(&m_ReferenceDataForIterativeCalib.first);
      r3->SetReferencePoints(m_ReferenceDataForIterativeCalib.second);
      r3->SetMaxAreaInPixels(maxArea);
      r3->SetUseContours(true);
      r3->SetUseInternalResampling(false);
      r3->SetUseTemplateMatching(false);
      r3->SetCaching(false);
      r3->SetInitialGuess(points);

      std::unique_ptr<niftk::PointDetector> l4(l3.release());
      std::unique_ptr<niftk::PointDetector> r4(r3.release());

      std::unique_ptr<niftk::SideBySideDetector> s3(new niftk::SideBySideDetector(l4, r4));
      s3->SetImage(&copyOfImage2);
      s3->SetImageScaleFactor(scaleFactors, doRescaleAfterPointExtraction);

      std::unique_ptr<niftk::PointDetector> s4(s3.release());

      std::shared_ptr<niftk::IPoint2DDetector> warpedDetector(s4.release());

      m_ImagesForWarping[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(warpedDetector, copyOfImage2));

      dynamic_cast<niftk::SideBySideDetector*>(
        m_ImagesForWarping[imageIndex].back().first.get())->SetImage(&(m_ImagesForWarping[imageIndex].back().second));
    }
  }
  else if (m_CalibrationPattern == TEMPLATE_MATCHING_NON_COPLANAR_RINGS)
  {
    cv::Size2i internalCorners(m_GridSizeX, m_GridSizeY);
    cv::Size2i offsetIfNotIterative(m_TemplateImage.cols / offsetForTemplateMatching,
                                    m_TemplateImage.rows / offsetForTemplateMatching);
    unsigned long int maxArea = m_TemplateImage.cols * m_TemplateImage.rows;

    // We set no scaleFactors, as this method uses a two phase
    // decorator pattern for the Tsai's method.
    // So, the SideBySide detector uses the scale factors
    // to scale up the image, as appropriate, so that each
    // point detector already has a scaled image.
    // So, each point detector requires no scale factors.
    cv::Point2d noScaleFactors(1, 1);

    std::unique_ptr<niftk::TemplateRingsPointDetector> l1(
      new niftk::TemplateRingsPointDetector(internalCorners,
                                            offsetIfNotIterative,
                                            cv::CALIB_CB_ASYMMETRIC_GRID
                                            | clusteringFlag
                                            ));

    l1->SetImageScaleFactor(noScaleFactors);
    l1->SetTemplateImage(&m_TemplateImage);
    l1->SetReferenceImage(&m_ReferenceDataForIterativeCalib.first);
    l1->SetReferencePoints(m_ReferenceDataForIterativeCalib.second);
    l1->SetMaxAreaInPixels(maxArea);
    l1->SetUseContours(true);
    l1->SetUseInternalResampling(true);
    l1->SetUseTemplateMatching(true);
    l1->SetCaching(true);

    std::unique_ptr<niftk::TemplateRingsPointDetector> r1(
      new niftk::TemplateRingsPointDetector(internalCorners,
                                            offsetIfNotIterative,
                                            cv::CALIB_CB_ASYMMETRIC_GRID
                                            | clusteringFlag
                                            ));

    r1->SetImageScaleFactor(noScaleFactors);
    r1->SetTemplateImage(&m_TemplateImage);
    r1->SetReferenceImage(&m_ReferenceDataForIterativeCalib.first);
    r1->SetReferencePoints(m_ReferenceDataForIterativeCalib.second);
    r1->SetMaxAreaInPixels(maxArea);
    r1->SetUseContours(true);
    r1->SetUseInternalResampling(true);
    r1->SetUseTemplateMatching(true);
    r1->SetCaching(true);

    std::unique_ptr<niftk::PointDetector> l2(l1.release());
    std::unique_ptr<niftk::PointDetector> r2(r1.release());

    std::unique_ptr<niftk::SideBySideDetector> s1(new niftk::SideBySideDetector(l2, r2));
    s1->SetImage(&copyOfImage1);
    s1->SetImageScaleFactor(scaleFactors, doRescaleAfterPointExtraction);

    std::unique_ptr<niftk::PointDetector> s2(s1.release());

    niftk::PointSet points = s2->GetPoints();

    if (points.size() == 2 * m_GridSizeX * m_GridSizeY)
    {
      isSuccessful = true;
      m_Points[imageIndex].push_back(points);

      std::shared_ptr<niftk::IPoint2DDetector> originalDetector(s2.release());

      m_OriginalImages[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(originalDetector, copyOfImage1));

      dynamic_cast<niftk::SideBySideDetector*>(
        m_OriginalImages[imageIndex].back().first.get())->SetImage(&(m_OriginalImages[imageIndex].back().second));

      std::unique_ptr<niftk::TemplateRingsPointDetector> l3(
        new niftk::TemplateRingsPointDetector(internalCorners,
                                              offsetIfNotIterative,
                                              cv::CALIB_CB_ASYMMETRIC_GRID
                                              | clusteringFlag
                                              ));

      l3->SetImageScaleFactor(noScaleFactors);
      l3->SetTemplateImage(&m_TemplateImage);
      l3->SetReferenceImage(&m_ReferenceDataForIterativeCalib.first);
      l3->SetReferencePoints(m_ReferenceDataForIterativeCalib.second);
      l3->SetMaxAreaInPixels(maxArea);
      l3->SetUseContours(true);
      l3->SetUseInternalResampling(false);
      l3->SetUseTemplateMatching(false);
      l3->SetCaching(false);
      l3->SetInitialGuess(points);

      std::unique_ptr<niftk::TemplateRingsPointDetector> r3(
        new niftk::TemplateRingsPointDetector(internalCorners,
                                              offsetIfNotIterative,
                                              cv::CALIB_CB_ASYMMETRIC_GRID
                                              | clusteringFlag
                                              ));

      r3->SetImageScaleFactor(noScaleFactors);
      r3->SetTemplateImage(&m_TemplateImage);
      r3->SetReferenceImage(&m_ReferenceDataForIterativeCalib.first);
      r3->SetReferencePoints(m_ReferenceDataForIterativeCalib.second);
      r3->SetMaxAreaInPixels(maxArea);
      r3->SetUseContours(true);
      r3->SetUseInternalResampling(false);
      r3->SetUseTemplateMatching(false);
      r3->SetCaching(false);
      r3->SetInitialGuess(points);

      std::unique_ptr<niftk::PointDetector> l4(l3.release());
      std::unique_ptr<niftk::PointDetector> r4(r3.release());

      std::unique_ptr<niftk::SideBySideDetector> s3(new niftk::SideBySideDetector(l4, r4));
      s3->SetImage(&copyOfImage2);
      s3->SetImageScaleFactor(scaleFactors, doRescaleAfterPointExtraction);

      std::unique_ptr<niftk::PointDetector> s4(s3.release());

      std::shared_ptr<niftk::IPoint2DDetector> warpedDetector(s4.release());

      m_ImagesForWarping[imageIndex].push_back(
        std::pair<std::shared_ptr<niftk::IPoint2DDetector>, cv::Mat>(warpedDetector, copyOfImage2));

      dynamic_cast<niftk::SideBySideDetector*>(
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

  if (m_ImageNode[0].IsNotNull() && m_ImageNode[1].IsNull())
  {
    // mono case, early exit.
    if (!extracted[0])
    {
      return isSuccessful;
    }
  }
  else
  {
    // stereo case, early exit.
    if (!extracted[0] && !extracted[1])
    {
      return isSuccessful;
    }
    if (extracted[0] && !extracted[1])
    {
      m_OriginalImages[0].pop_back();
      m_ImagesForWarping[0].pop_back();
      m_Points[0].pop_back();
      return isSuccessful;
    }
    if (!extracted[0] && extracted[1])
    {
      m_OriginalImages[1].pop_back();
      m_ImagesForWarping[1].pop_back();
      m_Points[1].pop_back();
      return isSuccessful;
    }
  }

  // Now we extract the tracking node.
  if (m_TrackingTransformNode.IsNotNull())
  {
    CoordinateAxesData::Pointer tracking = dynamic_cast<CoordinateAxesData*>(
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

    // Visualise it.
    CoordinateAxesData::Pointer copyOfTrackingMatrix = CoordinateAxesData::New();
    copyOfTrackingMatrix->SetVtkMatrix(*mat);
    mitk::DataNode::Pointer trackingNode = mitk::DataNode::New();
    trackingNode->SetData(copyOfTrackingMatrix);
    trackingNode->SetName("CalibrationTrackingData");
    trackingNode->SetVisibility(true);
    trackingNode->SetBoolProperty("helper object", true);
    trackingNode->SetBoolProperty("includeInBoundingBox", true);
    trackingNode->SetBoolProperty("show text", true);
    trackingNode->SetIntProperty("size", 100);

    m_TrackingMatricesDataNodes.push_back(trackingNode);
    if (m_DataStorage.IsNotNull())
    {
      m_DataStorage->Add(trackingNode);
    }

    // Finished.
    extracted[2] = true;
  }

  // Now we extract the model tracking node - ToDo: fix code duplication.
  if (m_ModelTransformNode.IsNotNull())
  {
    CoordinateAxesData::Pointer tracking = dynamic_cast<CoordinateAxesData*>(
          m_ModelTransformNode->GetData());

    if (tracking.IsNull())
    {
      mitkThrow() << "Model (e.g. chessboard) tracking node contains null tracking matrix.";
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
    m_ModelTrackingMatrices.push_back(openCVMat);
    extracted[3] = true;
  }

  // Then we check if we got everything, and therefore we are successful.
  if (extracted[0] // must always do left.
      && ((m_ImageNode[1].IsNotNull() && extracted[1]) // right image is optional
          || m_ImageNode[1].IsNull())
      && ((m_TrackingTransformNode.IsNotNull() && extracted[2]) // tracking node is optional
          || m_TrackingTransformNode.IsNull())
      && ((m_ModelTransformNode.IsNotNull() && extracted[3]) // reference tracking node is optional
          || m_ModelTransformNode.IsNull())
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
  if (m_Points[0].empty())
  {
    return;
  }

  for (int i = 0; i < 2; i++)
  {
    if (!m_Points[i].empty())
    {
      m_Points[i].pop_back();
      m_OriginalImages[i].pop_back();
      m_ImagesForWarping[i].pop_back();
    }
  }
  if (!m_TrackingMatricesDataNodes.empty())
  {
    if (m_DataStorage.IsNotNull())
    {
      m_DataStorage->Remove(m_TrackingMatricesDataNodes[m_TrackingMatricesDataNodes.size() - 1]);
    }
    m_TrackingMatricesDataNodes.pop_back();
    m_TrackingMatrices.pop_back();
  }
  if (!m_ModelTrackingMatrices.empty())
  {
    m_ModelTrackingMatrices.pop_back();
  }

  MITK_INFO << "UnGrab. Left point size now:" << m_Points[0].size() << ", right:" <<  m_Points[1].size();
}


//-----------------------------------------------------------------------------
bool NiftyCalVideoCalibrationManager::isStereo() const
{
  bool result = false;
  if (m_ImageNode[0].IsNotNull() && m_ImageNode[1].IsNotNull())
  {
    result = true;
  }
  return result;
}


//-----------------------------------------------------------------------------
double NiftyCalVideoCalibrationManager::GetMonoRMSReconstructionError(const cv::Matx44d& handEye)
{
  std::list<cv::Matx44d> trackingMatrices = this->ExtractTrackingMatrices();

  double rms = niftk::ComputeRMSReconstructionError(m_ModelPoints,
                                                    m_Points[0],
                                                    m_Rvecs[0],
                                                    m_Tvecs[0],
                                                    trackingMatrices,
                                                    handEye,
                                                    m_ModelToWorld
                                                   );
  return rms;
}


//-----------------------------------------------------------------------------
double NiftyCalVideoCalibrationManager::GetStereoRMSReconstructionError(const cv::Matx44d& handEye)
{
  std::list<cv::Matx44d> trackingMatrices = this->ExtractTrackingMatrices();

  double recon = niftk::ComputeRMSReconstructionError(m_ModelPoints,
                                                      m_Points[0],
                                                      m_Points[1],
                                                      m_Intrinsic[0],
                                                      m_Distortion[0],
                                                      m_Rvecs[0],
                                                      m_Tvecs[0],
                                                      m_Intrinsic[1],
                                                      m_Distortion[1],
                                                      m_LeftToRightRotationMatrix,
                                                      m_LeftToRightTranslationVector,
                                                      trackingMatrices,
                                                      handEye,
                                                      m_ModelToWorld
                                                     );
  return recon;
}


//-----------------------------------------------------------------------------
std::string NiftyCalVideoCalibrationManager::Calibrate()
{
  double rms = 0;

  cv::Matx21d tmpRMS;
  tmpRMS(0, 0) = 0;
  tmpRMS(1, 0) = 0;

  cv::Point2d sensorDimensions;
  sensorDimensions.x = 1;
  sensorDimensions.y = 1;

  if (m_ImageNode[0].IsNull())
  {
    mitkThrow() << "Left image should never be NULL.";
  }

  if (m_ModelPoints.empty())
  {
    mitkThrow() << "Model should never be empty.";
  }

  cv::Size2i imageSize = m_ImageSize;
  if (m_NumberOfSnapshotsForCalibrating == 1) // i.e. must be doing Tsai.
  {
    imageSize.width = m_ImageSize.width * m_ScaleFactorX;
    imageSize.height = m_ImageSize.height * m_ScaleFactorY;
  }

  {
    std::ostringstream message;
    message << "Calibrating with " <<  m_NumberOfSnapshotsForCalibrating
            << " sample" << (m_NumberOfSnapshotsForCalibrating > 1 ? "s" : "")
            << std::endl;
    m_CalibrationResult = message.str();
  }

  if (m_DoIterative)
  {
    if (m_ImageNode[1].IsNull())
    {
      rms = niftk::IterativeMonoCameraCalibration(
        m_ModelPoints,
        m_ReferenceDataForIterativeCalib,
        m_OriginalImages[0],
        m_ImagesForWarping[0],
        imageSize,
        m_Intrinsic[0],
        m_Distortion[0],
        m_Rvecs[0],
        m_Tvecs[0]
       );

      {
        std::ostringstream message;
        message << "Iterative mono: " << rms << " pixels" << std::endl;
        m_CalibrationResult += message.str();
      }
    }
    else
    {
      tmpRMS = niftk::IterativeStereoCameraCalibration(
        m_ModelPoints,
        m_ReferenceDataForIterativeCalib,
        m_OriginalImages[0],
        m_OriginalImages[1],
        imageSize,
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
        m_LeftToRightRotationMatrix,
        m_LeftToRightTranslationVector,
        m_EssentialMatrix,
        m_FundamentalMatrix,
        0,
        m_Do3DOptimisation
        );

      {
        std::ostringstream message;
        message << "Iterative Stereo: " << tmpRMS(0,0) << " pixels" << std::endl;
        message << "Iterative Stereo: " << tmpRMS(1, 0) << " mm" << std::endl;
        m_CalibrationResult += message.str();
      }
    }
  }
  else
  {
    if (m_Points[0].size() == 1)
    {
      cv::Mat rvecLeft;
      cv::Mat tvecLeft;

      rms = niftk::TsaiMonoCameraCalibration(m_ModelPoints,
                                             *(m_Points[0].begin()),
                                             imageSize,
                                             sensorDimensions,
                                             m_Intrinsic[0],
                                             m_Distortion[0],
                                             rvecLeft,
                                             tvecLeft
                                            );

      {
        std::ostringstream message;
        message << "Tsai mono left: " << rms << " pixels" << std::endl;
        m_CalibrationResult += message.str();
      }

      m_Rvecs[0].clear();
      m_Tvecs[0].clear();

      m_Rvecs[0].push_back(rvecLeft);
      m_Tvecs[0].push_back(tvecLeft);
    }
    else
    {
      rms = niftk::ZhangMonoCameraCalibration(
        m_ModelPoints,
        m_Points[0],
        imageSize,
        m_Intrinsic[0],
        m_Distortion[0],
        m_Rvecs[0],
        m_Tvecs[0]
        );

      {
        std::ostringstream message;
        message << "Zhang mono left: " << rms << " pixels" << std::endl;
        m_CalibrationResult += message.str();
      }

    }

    if (m_ImageNode[1].IsNotNull())
    {
      if (m_Points[1].size() == 1)
      {
        cv::Mat rvecRight;
        cv::Mat tvecRight;

        rms = niftk::TsaiMonoCameraCalibration(m_ModelPoints,
                                               *(m_Points[1].begin()),
                                               imageSize,
                                               sensorDimensions,
                                               m_Intrinsic[1],
                                               m_Distortion[1],
                                               rvecRight,
                                               tvecRight
                                              );

        {
          std::ostringstream message;
          message << "Tsai mono right: " << rms << " pixels" << std::endl;
          m_CalibrationResult += message.str();
        }

        m_Rvecs[1].clear();
        m_Tvecs[1].clear();

        m_Rvecs[1].push_back(rvecRight);
        m_Tvecs[1].push_back(tvecRight);
      }
      else
      {
        rms = niftk::ZhangMonoCameraCalibration(
          m_ModelPoints,
          m_Points[1],
          imageSize,
          m_Intrinsic[1],
          m_Distortion[1],
          m_Rvecs[1],
          m_Tvecs[1]
          );

        {
          std::ostringstream message;
          message << "Zhang mono right: " << rms << " pixels" << std::endl;
          m_CalibrationResult += message.str();
        }
      }

      tmpRMS = niftk::StereoCameraCalibration(
        m_ModelPoints,
        m_Points[0],
        m_Points[1],
        imageSize,
        m_Intrinsic[0],
        m_Distortion[0],
        m_Rvecs[0],
        m_Tvecs[0],
        m_Intrinsic[1],
        m_Distortion[1],
        m_Rvecs[1],
        m_Tvecs[1],
        m_LeftToRightRotationMatrix,
        m_LeftToRightTranslationVector,
        m_EssentialMatrix,
        m_FundamentalMatrix,
        CV_CALIB_USE_INTRINSIC_GUESS | CV_CALIB_FIX_INTRINSIC,
        m_Do3DOptimisation
        );
      rms = tmpRMS(1, 0);

      {
        std::ostringstream message;
        message << "Stereo: " << tmpRMS(0,0) << " pixels" << std::endl;
        message << "Stereo: " << tmpRMS(1,0) << " mm" << std::endl;
        m_CalibrationResult += message.str();
      }
    }
  }

  // If we have tracking info, do all hand-eye methods .
  if (m_TrackingTransformNode.IsNotNull())
  {
    {
      std::ostringstream message;
      message << std::endl << "Calibrating hand-eye:" << std::endl;
      m_CalibrationResult += message.str();
    }

    // Don't change the order of these sections where we compute each hand-eye.
    if (m_TrackingMatrices.size() > 1)
    {
      m_HandEyeMatrices[0][TSAI_1989] = DoTsaiHandEye(0);
      {
        m_ModelToWorld = this->GetModelToWorld(m_HandEyeMatrices[0][TSAI_1989]);
        rms = this->GetMonoRMSReconstructionError(m_HandEyeMatrices[0][TSAI_1989]);
        std::ostringstream message;
        message << "Tsai mono left: " << rms << " mm" << std::endl;
        m_CalibrationResult += message.str();
      }
    }

    m_HandEyeMatrices[0][SHAHIDI_2002] = DoShahidiHandEye(0);
    {
      m_ModelToWorld = this->GetModelToWorld(m_HandEyeMatrices[0][SHAHIDI_2002]);
      rms = this->GetMonoRMSReconstructionError(m_HandEyeMatrices[0][SHAHIDI_2002]);
      std::ostringstream message;
      message << "Shahidi mono left: " << rms << " mm" << std::endl;
      m_CalibrationResult += message.str();
    }

    m_HandEyeMatrices[0][MALTI_2013] = DoMaltiHandEye(0);
    {
      m_ModelToWorld = this->GetModelToWorld(m_HandEyeMatrices[0][MALTI_2013]);
      rms = this->GetMonoRMSReconstructionError(m_HandEyeMatrices[0][MALTI_2013]);
      std::ostringstream message;
      message << "Malti mono left: " << rms << " mm" << std::endl;
      m_CalibrationResult += message.str();
    }

    m_HandEyeMatrices[0][NON_LINEAR_EXTRINSIC] = DoFullExtrinsicHandEye(0);
    {
      m_ModelToWorld = this->GetModelToWorld(m_HandEyeMatrices[0][NON_LINEAR_EXTRINSIC]);
      rms = this->GetMonoRMSReconstructionError(m_HandEyeMatrices[0][NON_LINEAR_EXTRINSIC]);
      std::ostringstream message;
      message << "Non-Linear Ext mono left: " << rms << " mm" << std::endl;
      m_CalibrationResult += message.str();
    }

    if (m_ImageNode[1].IsNotNull())
    {
      // Don't change the order of these sections where we compute each hand-eye.
      if (m_TrackingMatrices.size() > 1)
      {
        m_HandEyeMatrices[1][TSAI_1989] = DoTsaiHandEye(1);
        {
          m_ModelToWorld = this->GetModelToWorld(m_HandEyeMatrices[0][TSAI_1989]);
          rms = this->GetStereoRMSReconstructionError(m_HandEyeMatrices[0][TSAI_1989]);
          std::ostringstream message;
          message << "Tsai stereo: " << rms << " mm" << std::endl;
          m_CalibrationResult += message.str();
        }
      }
      m_HandEyeMatrices[1][SHAHIDI_2002] = DoShahidiHandEye(1);
      {
        m_ModelToWorld = this->GetModelToWorld(m_HandEyeMatrices[0][SHAHIDI_2002]);
        rms = this->GetStereoRMSReconstructionError(m_HandEyeMatrices[0][SHAHIDI_2002]);
        std::ostringstream message;
        message << "Shahidi stereo: " << rms << " mm" << std::endl;
        m_CalibrationResult += message.str();
      }
      m_HandEyeMatrices[1][MALTI_2013] = DoMaltiHandEye(1);
      {
        m_ModelToWorld = this->GetModelToWorld(m_HandEyeMatrices[0][MALTI_2013]);
        rms = this->GetStereoRMSReconstructionError(m_HandEyeMatrices[0][MALTI_2013]);
        std::ostringstream message;
        message << "Malti stereo: " << rms << " mm" << std::endl;
        m_CalibrationResult += message.str();
      }
      DoFullExtrinsicHandEyeInStereo(m_HandEyeMatrices[0][NON_LINEAR_EXTRINSIC],
                                     m_HandEyeMatrices[1][NON_LINEAR_EXTRINSIC]
                                    );
      {
        m_ModelToWorld = this->GetModelToWorld(m_HandEyeMatrices[0][NON_LINEAR_EXTRINSIC]);
        rms = this->GetStereoRMSReconstructionError(m_HandEyeMatrices[0][NON_LINEAR_EXTRINSIC]);
        std::ostringstream message;
        message << "Non-Linear Ext stereo: " << rms << " mm" << std::endl;
        m_CalibrationResult += message.str();
      }
    } // end if we are in stereo.

    // This is so that the one we see on screen is our prefered one.
    m_ModelToWorld = this->GetModelToWorld(m_HandEyeMatrices[0][m_HandeyeMethod]);

  } // end if we have tracking data.

  // Sets properties on images.
  this->UpdateDisplayNodes();

  MITK_INFO << m_CalibrationResult;
  return m_CalibrationResult;
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::UpdateDisplayNodes()
{
  // If true, we set properties on the images,
  // so that the overlay viewer renders in correctly calibrated mode.
  if (m_UpdateNodes)
  {
    if (m_ImageNode[0].IsNotNull())
    {
      this->SetIntrinsicsOnImage(m_Intrinsic[0], m_Distortion[0], "niftk.CameraCalibration", m_ImageNode[0]);
    }

    if (m_ImageNode[1].IsNotNull())
    {
      this->SetIntrinsicsOnImage(m_Intrinsic[1],
                                 m_Distortion[1],
                                 "niftk.CameraCalibration",
                                 m_ImageNode[1]
                                );

      this->SetStereoExtrinsicsOnImage(m_LeftToRightRotationMatrix,
                                       m_LeftToRightTranslationVector,
                                       "niftk.StereoRigTransformation",
                                       m_ImageNode[1]
                                      );
    } // end if right hand image
  } // end if updating nodes
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::SetIntrinsicsOnImage(const cv::Mat& intrinsics,
                                                           const cv::Mat& distortion,
                                                           const std::string& propertyName,
                                                           mitk::DataNode::Pointer imageNode)
{
  mitk::CameraIntrinsics::Pointer intrinsicsHolder = mitk::CameraIntrinsics::New();
  intrinsicsHolder->SetIntrinsics(intrinsics, distortion);

  mitk::CameraIntrinsicsProperty::Pointer intrinsicsProperty = mitk::CameraIntrinsicsProperty::New(intrinsicsHolder);
  imageNode->SetProperty(propertyName.c_str(), intrinsicsProperty);

  mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(imageNode->GetData());
  if (image.IsNotNull())
  {
    image->SetProperty(propertyName.c_str(), intrinsicsProperty);
  }
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::SetStereoExtrinsicsOnImage(const cv::Mat& leftToRightRotationMatrix,
                                                                 const cv::Mat& leftToRightTranslationVector,
                                                                 const std::string& propertyName,
                                                                 mitk::DataNode::Pointer imageNode
                                                                )
{
  cv::Matx44d leftToRight = niftk::RotationAndTranslationToMatrix(
        leftToRightRotationMatrix, leftToRightTranslationVector);

  cv::Matx44d rightToLeft = leftToRight.inv();
  itk::Matrix<float, 4, 4>    txf;
  txf.SetIdentity();
  for (int r = 0; r < 4; r++)
  {
    for (int c = 0; c < 4; c++)
    {
      txf.GetVnlMatrix()(r, c) = rightToLeft(r, c);
    }
  }

  niftk::Undistortion::MatrixProperty::Pointer matrixProp = niftk::Undistortion::MatrixProperty::New(txf);
  imageNode->SetProperty(propertyName.c_str(), matrixProp);

  mitk::Image::Pointer image = dynamic_cast<mitk::Image*>(imageNode->GetData());
  if (image.IsNotNull())
  {
    image->SetProperty(propertyName.c_str(), matrixProp);
  }
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::LoadCalibrationFromDirectory(const std::string& dirName)
{
  cv::Mat leftCameraIntrinsic = cv::Mat(3,3,CV_64FC1);
  cv::Mat leftCameraDistortion = cv::Mat(1,4,CV_64FC1); // we only save 4 params, as MITK only uses 4.
  cv::Mat rightCameraIntrinsic = cv::Mat(3,3,CV_64FC1);
  cv::Mat rightCameraDistortion = cv::Mat(1,4,CV_64FC1); // we only save 4 params, as MITK only uses 4.
  cv::Mat rightToLeftRotationMatrix = cv::Mat(3,3,CV_64FC1);
  cv::Mat rightToLeftTranslationVector = cv::Mat(3,1,CV_64FC1);
  cv::Mat leftCameraToTracker = cv::Mat(4,4,CV_64FC1);
  cv::Mat rightCameraToTracker = cv::Mat(4,4,CV_64FC1);
  cv::Mat modelToWorld = cv::Mat(4,4,CV_64FC1);

  std::string dir = dirName + niftk::GetFileSeparator();

  std::string leftIntrinsicsFile  = dir + "calib.left.intrinsic.txt";
  std::string rightIntrinsicsFile = dir + "calib.right.intrinsic.txt";
  std::string rightToLeftFile     = dir + "calib.r2l.txt";
  std::string leftEyeHandFile     = dir + "calib.left.eyehand.current.txt";
  std::string rightEyeHandFile    = dir + "calib.right.eyehand.current.txt";
  std::string modelToWorldFile    = dir + "calib.model2world.txt";

  mitk::LoadCameraIntrinsicsFromPlainText(leftIntrinsicsFile,
                                          &leftCameraIntrinsic, &leftCameraDistortion);

  mitk::LoadCameraIntrinsicsFromPlainText(rightIntrinsicsFile,
                                          &rightCameraIntrinsic, &rightCameraDistortion);

  mitk::LoadStereoTransformsFromPlainText(rightToLeftFile,
                                          &rightToLeftRotationMatrix, &rightToLeftTranslationVector);

  mitk::LoadHandeyeFromPlainText(leftEyeHandFile,
                                 &leftCameraToTracker);

  mitk::LoadHandeyeFromPlainText(rightEyeHandFile,
                                 &rightCameraToTracker);

  mitk::LoadHandeyeFromPlainText(modelToWorldFile,
                                 &modelToWorld);

  cv::Matx44d rightToLeft = niftk::RotationAndTranslationToMatrix(rightToLeftRotationMatrix,
                                                                  rightToLeftTranslationVector
                                                                  );
  cv::Matx44d leftToRight = rightToLeft.inv();

  // Observe: All code in NiftyCal uses 'Hand-Eye', and is consistent.
  // In NifTK, the 4x4 'hand-eye' matrix we save to disk is in fact an 'eye-hand'.
  // So, see here how we are calling it first a camera-to-tracker which is analagous to eye-hand.
  // We then invert it, so in this class, anything called hand-eye really is a hand-eye.

  cv::Matx44d leftEyeHand(leftCameraToTracker);
  cv::Matx44d leftHandEye = leftEyeHand.inv(cv::DECOMP_SVD);
  cv::Matx44d rightEyeHand(rightCameraToTracker);
  cv::Matx44d rightHandEye = rightEyeHand.inv(cv::DECOMP_SVD);

  for (int r = 0; r < 3; r++)
  {
    for (int c = 0; c < 3; c++)
    {
      m_LeftToRightRotationMatrix.at<double>(r, c) = leftToRight(r, c);
      m_Intrinsic[0].at<double>(r, c) = leftCameraIntrinsic.at<double>(r, c);
      m_Intrinsic[1].at<double>(r, c) = rightCameraIntrinsic.at<double>(r, c);
      m_ModelToWorld(r, c) = modelToWorld.at<double>(r,c);
    }
    m_LeftToRightTranslationVector.at<double>(0, r) = leftToRight(r, 3);
    m_ModelToWorld(r, 3) = modelToWorld.at<double>(r, 3);
  }
  for (int i = 0; i < 4; i++)
  {
    m_HandEyeMatrices[0][i] = leftHandEye;
    m_HandEyeMatrices[1][i] = rightHandEye;
    m_Distortion[0].at<double>(0, i) = leftCameraDistortion.at<double>(0, i);
    m_Distortion[1].at<double>(0, i) = rightCameraDistortion.at<double>(0, i);
  }

  m_CalibrationDirName = dirName;
  this->UpdateDisplayNodes();
}


//-----------------------------------------------------------------------------
void NiftyCalVideoCalibrationManager::Save()
{
  if (m_ImageNode[0].IsNull())
  {
    mitkThrow() << "Left image should never be NULL.";
  }

  if (m_OutputPrefixName.empty())
  {
    mitkThrow() << "Empty output directory name.";
  }

  niftk::SystemTimeServiceRAII timeService;
  niftk::SystemTimeServiceI::TimeType timeInNanoseconds = timeService.GetSystemTimeInNanoseconds();
  niftk::SystemTimeServiceI::TimeType timeInMilliseconds = timeInNanoseconds/1000000;
  std::string formattedTime = FormatDateTimeAsStdString(timeInMilliseconds);

  std::ostringstream dirName;
  dirName << m_OutputPrefixName << niftk::GetFileSeparator() << formattedTime << niftk::GetFileSeparator();
  m_OutputDirName = dirName.str();

  MITK_INFO << "Saving calibration to:" << m_OutputDirName << ":";

  if (!niftk::CreateDirAndParents(m_OutputDirName))
  {
    mitkThrow() << "Failed to create directory:" << m_OutputDirName;
  }

  niftk::SaveNifTKIntrinsics(m_Intrinsic[0], m_Distortion[0], m_OutputDirName + "calib.left.intrinsic.txt");
  this->SaveImages("calib.left.images.", m_OriginalImages[0]);
  this->SavePoints("calib.left.points.", m_Points[0]);

  if (m_ImageNode[1].IsNotNull())
  {
    niftk::SaveNifTKIntrinsics(
      m_Intrinsic[1], m_Distortion[1], m_OutputDirName + "calib.right.intrinsic.txt");

    niftk::SaveNifTKStereoExtrinsics(
      m_LeftToRightRotationMatrix, m_LeftToRightTranslationVector, m_OutputDirName + "calib.r2l.txt");

    this->SaveImages("calib.right.images.", m_OriginalImages[1]);
    this->SavePoints("calib.right.points.", m_Points[1]);
  }

  if (m_ImageNode[0].IsNotNull())
  {
    int counter = 0;
    std::list<cv::Matx44d> leftCams = this->ExtractCameraMatrices(0);
    std::list<cv::Matx44d >::const_iterator iter;
    for (iter = leftCams.begin();
         iter != leftCams.end();
         ++iter
         )
    {
      std::ostringstream fileName;
      fileName << m_OutputDirName << "calib.left.camera." << counter++ << ".4x4";
      niftk::Save4x4Matrix(*iter, fileName.str());
    }
  }

  if (m_ImageNode[1].IsNotNull())
  {
    int counter = 0;
    std::list<cv::Matx44d> rightCams = this->ExtractCameraMatrices(1);
    std::list<cv::Matx44d >::const_iterator iter;
    for (iter = rightCams.begin();
         iter != rightCams.end();
         ++iter
         )
    {
      std::ostringstream fileName;
      fileName << m_OutputDirName << "calib.right.camera." << counter++ << ".4x4";
      niftk::Save4x4Matrix(*iter, fileName.str());
    }
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
    niftk::Save4x4Matrix(m_HandEyeMatrices[0][0].inv(), m_OutputDirName
        + "calib.left.eyehand.tsai.txt");
    niftk::Save4x4Matrix(m_HandEyeMatrices[0][1].inv(), m_OutputDirName
        + "calib.left.eyehand.shahidi.txt");
    niftk::Save4x4Matrix(m_HandEyeMatrices[0][2].inv(), m_OutputDirName
        + "calib.left.eyehand.malti.txt");
    niftk::Save4x4Matrix(m_HandEyeMatrices[0][3].inv(), m_OutputDirName
        + "calib.left.eyehand.allextrinsic.txt");
    niftk::Save4x4Matrix(m_HandEyeMatrices[0][m_HandeyeMethod].inv(), m_OutputDirName
        + "calib.left.eyehand.current.txt");

    niftk::SaveRigidParams(m_HandEyeMatrices[0][0].inv(), m_OutputDirName
        + "calib.left.eyehand.tsai.params.txt");
    niftk::SaveRigidParams(m_HandEyeMatrices[0][1].inv(), m_OutputDirName
        + "calib.left.eyehand.shahidi.params.txt");
    niftk::SaveRigidParams(m_HandEyeMatrices[0][2].inv(), m_OutputDirName
        + "calib.left.eyehand.malti.params.txt");
    niftk::SaveRigidParams(m_HandEyeMatrices[0][3].inv(), m_OutputDirName
        + "calib.left.eyehand.allextrinsic.params.txt");
    niftk::SaveRigidParams(m_HandEyeMatrices[0][m_HandeyeMethod].inv(), m_OutputDirName
        + "calib.left.eyehand.current.params.txt");

    if (m_ImageNode[1].IsNotNull())
    {
      // We deliberately output all hand-eye matrices, and additionally, whichever one was preferred method.
      niftk::Save4x4Matrix(m_HandEyeMatrices[1][0].inv(), m_OutputDirName
          + "calib.right.eyehand.tsai.txt");
      niftk::Save4x4Matrix(m_HandEyeMatrices[1][1].inv(), m_OutputDirName
          + "calib.right.eyehand.shahidi.txt");
      niftk::Save4x4Matrix(m_HandEyeMatrices[1][2].inv(), m_OutputDirName
          + "calib.right.eyehand.malti.txt");
      niftk::Save4x4Matrix(m_HandEyeMatrices[1][3].inv(), m_OutputDirName
          + "calib.right.eyehand.allextrinsic.txt");
      niftk::Save4x4Matrix(m_HandEyeMatrices[1][m_HandeyeMethod].inv(), m_OutputDirName
          + "calib.right.eyehand.current.txt");

      niftk::SaveRigidParams(m_HandEyeMatrices[1][0].inv(), m_OutputDirName
          + "calib.right.eyehand.tsai.params.txt");
      niftk::SaveRigidParams(m_HandEyeMatrices[1][1].inv(), m_OutputDirName
          + "calib.right.eyehand.shahidi.params.txt");
      niftk::SaveRigidParams(m_HandEyeMatrices[1][2].inv(), m_OutputDirName
          + "calib.right.eyehand.malti.params.txt");
      niftk::SaveRigidParams(m_HandEyeMatrices[1][3].inv(), m_OutputDirName
          + "calib.right.eyehand.allextrinsic.params.txt");
      niftk::SaveRigidParams(m_HandEyeMatrices[1][m_HandeyeMethod].inv(), m_OutputDirName
          + "calib.right.eyehand.current.params.txt");
    }

    if (m_ModelTransformNode.IsNotNull())
    {
      int counter = 0;
      std::list<cv::Matx44d >::const_iterator iter;
      for (iter = m_ModelTrackingMatrices.begin();
           iter != m_ModelTrackingMatrices.end();
           ++iter
           )
      {
        std::ostringstream fileName;
        fileName << m_OutputDirName << "calib.tracking.model." << counter++ << ".4x4";
        niftk::Save4x4Matrix(*iter, fileName.str());
      }
    } // end if we have a reference transform

    niftk::Save4x4Matrix(m_ModelToWorld, m_OutputDirName + "calib.model2world.txt");

  } // end if we have tracking info

  // Write main results to file.
  std::string outputMessageFileName = m_OutputDirName + "calib.result.log";
  std::ofstream outputMessageFile;
  outputMessageFile.open (outputMessageFileName, std::ofstream::out);
  if (!outputMessageFile.is_open())
  {
    mitkThrow() << "Failed to open file:" << outputMessageFileName << " for writing.";
  }
  outputMessageFile << m_CalibrationResult << std::endl;
  outputMessageFile.close();

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

    // We currently convert all input images to grey when we grab.
    // So for now, just convert back to 3 channel image.
    // This means, that when we save a load of images,
    // we can load them back in and run camera calibration and undistortion manually.
    //
    // Otherwise - we would need to do more extensive refactoring.
    // TODO: Tidy this up.
    cv::Mat tmp;
    cv::cvtColor((*iter).second, tmp, CV_GRAY2RGB);

    cv::imwrite(fileName.str(), tmp);
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

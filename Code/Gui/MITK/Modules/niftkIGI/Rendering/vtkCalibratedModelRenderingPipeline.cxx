/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "vtkCalibratedModelRenderingPipeline.h"
#include <mitkExceptionMacro.h>
#include <niftkFileHelper.h>
#include <niftkVTKFunctions.h>
#include <mitkOpenCVMaths.h>
#include <mitkCameraIntrinsicsProperty.h>
#include <mitkCameraCalibrationFacade.h>

//-----------------------------------------------------------------------------
vtkCalibratedModelRenderingPipeline::vtkCalibratedModelRenderingPipeline(
    const std::string& name,
    const mitk::Point2I& windowSize,
    const mitk::Point2I& calibratedWindowSize,
    const std::string& leftIntrinsicsFileName,
    const std::string& rightIntrinsicsFileName,
    const std::string& visualisationModelFileName,
    const std::string& rightToLeftFileName,
    const std::string& textureFileName,
    const std::string& trackingModelFileName,
    const float& trackingGlyphRadius
    )
  : m_Name(name), m_UseDistortion(false), m_IsRightHandCamera(false)
{
  // Do all validation early, and bail out without doing anything.
  if (windowSize[0] <= 0 || windowSize[1] <= 0)
  {
    mitkThrow() << "Invalid windowSize:" << windowSize;
  }
  if (calibratedWindowSize[0] <= 0 || calibratedWindowSize[1] <= 0)
  {
    mitkThrow() << "Invalid windowSize:" << windowSize;
  }
  if (!niftk::FileExists(leftIntrinsicsFileName))
  {
    mitkThrow() << "Left Intrinsics file does not exist:" << leftIntrinsicsFileName;
  }
  if (!niftk::FileExists(rightIntrinsicsFileName))
  {
    mitkThrow() << "Right Intrinsics file does not exist:" << rightIntrinsicsFileName;
  }
  if (!niftk::FileExists(visualisationModelFileName))
  {
    mitkThrow() << "Model does not exist:" << visualisationModelFileName;
  }
  if (rightToLeftFileName.length() > 0 && !niftk::FileExists(rightToLeftFileName))
  {
    mitkThrow() << "Right to left file name is specified but doesn't exist:" << rightToLeftFileName;
  }
  if (textureFileName.length() > 0 && !niftk::FileExists(textureFileName))
  {
    mitkThrow() << "Texture file name is specified but doesn't exist:" << textureFileName;
  }
  if (trackingModelFileName.length() > 0 && !niftk::FileExists(trackingModelFileName))
  {
    mitkThrow() << "Tracking model file name is specified but doesn't exist:" << trackingModelFileName;
  }

  // Loading intrinsics.
  m_LeftIntrinsicMatrix = cvCreateMat (3,3,CV_64FC1);
  m_LeftDistortionVector = cvCreateMat (1,4,CV_64FC1);
  mitk::LoadCameraIntrinsicsFromPlainText(leftIntrinsicsFileName, &m_LeftIntrinsicMatrix, &m_LeftDistortionVector);

  m_RightIntrinsicMatrix = cvCreateMat (3,3,CV_64FC1);
  m_RightDistortionVector = cvCreateMat (1,4,CV_64FC1);
  if (rightIntrinsicsFileName.length() > 0 && niftk::FileExists(rightIntrinsicsFileName))
  {
    mitk::LoadCameraIntrinsicsFromPlainText(rightIntrinsicsFileName, &m_RightIntrinsicMatrix, &m_RightDistortionVector);
  }

  // Loading Right-To-Left if it was provided.
  cv::Mat rightToLeftRotationMatrix = cvCreateMat (3,3,CV_64FC1);
  cv::Mat rightToLeftTranslationVector = cvCreateMat (1,3,CV_64FC1);
  m_CameraMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  m_CameraMatrixInverted = vtkSmartPointer<vtkMatrix4x4>::New();
  m_RightToLeftMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  m_RightToLeftMatrix->Identity();
  if (rightToLeftFileName.length() > 0 && niftk::FileExists(rightToLeftFileName))
  {
    // This should throw exceptions on failure. Haven't checked yet.
    mitk::LoadStereoTransformsFromPlainText(rightToLeftFileName, &rightToLeftRotationMatrix, &rightToLeftTranslationVector);

    for (unsigned int i = 0; i < 3; i++)
    {
      for (unsigned int j = 0; j < 3; j++)
      {
        m_RightToLeftMatrix->SetElement(i,j, rightToLeftRotationMatrix.at<double>(i, j));
      }
      m_RightToLeftMatrix->SetElement(i,3, rightToLeftTranslationVector.at<double>(0, i));
    }
  }

  mitk::Point2D aspect;
  aspect[0] = static_cast<double>(windowSize[0])/static_cast<double>(calibratedWindowSize[0]);
  aspect[1] = static_cast<double>(windowSize[1])/static_cast<double>(calibratedWindowSize[1]);

  m_Camera = vtkSmartPointer<vtkOpenGLMatrixDrivenCamera>::New();
  m_Camera->SetActualWindowSize(windowSize[0], windowSize[1]);
  m_Camera->SetCalibratedImageSize(calibratedWindowSize[0], calibratedWindowSize[1], aspect[0]/aspect[1]);
  m_Camera->SetIntrinsicParameters(m_LeftIntrinsicMatrix.at<double>(0,0), m_LeftIntrinsicMatrix.at<double>(1,1), m_LeftIntrinsicMatrix.at<double>(0,2), m_LeftIntrinsicMatrix.at<double>(1,2));
  m_Camera->SetUseCalibratedCamera(true);

  // We transform all models into a global world coordinate system.
  m_ModelToWorldMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  m_ModelToWorldMatrix->Identity();
  m_ModelToWorldTransform = vtkSmartPointer<vtkMatrixToLinearTransform>::New();
  m_ModelToWorldTransform->SetInput(m_ModelToWorldMatrix);
  m_CameraToWorldMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  m_CameraToWorldMatrix->Identity();

  // Create the tracking pipeline. Its optional whether we update it.
  m_TrackingModelReader = vtkSmartPointer<vtkPolyDataReader>::New();
  m_TrackingModelTransformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
  m_TrackingModelTransformFilter->SetTransform(m_ModelToWorldTransform);
  m_TrackingModelWriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  m_TrackingModelWriter->SetInputConnection(m_TrackingModelTransformFilter->GetOutputPort());
  m_SphereForGlyph = vtkSmartPointer<vtkSphereSource>::New();
  m_SphereForGlyph->SetRadius(trackingGlyphRadius);
  m_GlyphFilter = vtkSmartPointer<vtkGlyph3D>::New();
  m_GlyphFilter->SetSourceData(m_SphereForGlyph->GetOutput());
  m_GlyphFilter->SetInputConnection(m_TrackingModelTransformFilter->GetOutputPort());
  m_GlyphFilter->SetScaleModeToDataScalingOff();
  m_TrackingModelMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  m_TrackingModelMapper->SetInputConnection(m_GlyphFilter->GetOutputPort());
  m_TrackingModelActor = vtkSmartPointer<vtkActor>::New();
  m_TrackingModelActor->SetMapper(m_TrackingModelMapper);
  m_TrackingModelActor->GetProperty()->BackfaceCullingOn();

  // If filename provided, load data early, and force one initialisation of pipeline.
  if (trackingModelFileName.length() > 0)
  {
    m_TrackingModelReader->SetFileName(trackingModelFileName.c_str());
    m_TrackingModelReader->Update();
    m_TrackingModelTransformFilter->SetInputConnection(m_TrackingModelReader->GetOutputPort());
    m_TrackingModelTransformFilter->Update();
    m_SphereForGlyph->Update();
    m_GlyphFilter->Update();
    m_TrackingModelMapper->Update();
  }

  // Create the visualisation pipeline.
  m_VisualisationModelReader = vtkSmartPointer<vtkPolyDataReader>::New();
  m_VisualisationModelTransformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
  m_VisualisationModelTransformFilter->SetInputConnection(m_VisualisationModelReader->GetOutputPort());
  m_VisualisationModelTransformFilter->SetTransform(m_ModelToWorldTransform);
  m_VisualisationModelWriter = vtkSmartPointer<vtkPolyDataWriter>::New();
  m_VisualisationModelWriter->SetInputConnection(m_VisualisationModelTransformFilter->GetOutputPort());
  m_TextureReader = vtkSmartPointer<vtkPNGReader>::New();
  m_Texture = vtkSmartPointer<vtkTexture>::New();
  m_Texture->SetInputConnection(m_TextureReader->GetOutputPort());
  m_Texture->InterpolateOff();
  m_VisualisationModelMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  m_VisualisationModelMapper->SetInputConnection(m_VisualisationModelTransformFilter->GetOutputPort());
  m_VisualisationModelMapper->ScalarVisibilityOff();
  m_VisualisationModelActor = vtkSmartPointer<vtkActor>::New();
  m_VisualisationModelActor->GetProperty()->BackfaceCullingOn();
  m_VisualisationModelActor->GetProperty()->SetInterpolationToFlat();

  // The visualisation model is not optional.
  m_VisualisationModelReader->SetFileName(visualisationModelFileName.c_str());
  m_VisualisationModelReader->Update();
  m_VisualisationModelTransformFilter->Update();
  m_VisualisationModelMapper->Update();
  m_VisualisationModelActor->SetMapper(m_VisualisationModelMapper);

  // The texture mapping is optional.
  if (textureFileName.c_str())
  {
    m_TextureReader->SetFileName(textureFileName.c_str());
    m_TextureReader->Update();
    m_VisualisationModelActor->SetTexture(m_Texture);
  }

  // This creates the render window.
  m_Renderer = vtkSmartPointer<vtkRenderer>::New();
  m_Renderer->SetBackground(0, 0, 255);  // RGB
  m_Renderer->AddActor(m_VisualisationModelActor);
  if (trackingModelFileName.length() > 0)
  {
    // Again, tracking model is optional.
    m_Renderer->AddActor(m_TrackingModelActor);
  }
  m_Renderer->SetActiveCamera(m_Camera);
  m_Renderer->SetLightFollowCamera(true);

  m_RenderWin = vtkSmartPointer<vtkRenderWindow>::New();
  m_RenderWin->AddRenderer(m_Renderer);
  m_RenderWin->SetSize(windowSize[0], windowSize[1]);
  m_RenderWin->SetWindowName(m_Name.c_str());
  m_RenderWin->DoubleBufferOff();

  // Force initialisation of position.
  vtkSmartPointer<vtkMatrix4x4> identityMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  identityMatrix->Identity();
  this->SetIsRightHandCamera(false);
  this->SetModelToWorldMatrix(*identityMatrix);
  this->SetCameraToWorldMatrix(*identityMatrix);
  this->Render();
}


//-----------------------------------------------------------------------------
vtkCalibratedModelRenderingPipeline::~vtkCalibratedModelRenderingPipeline()
{
}


//-----------------------------------------------------------------------------
vtkPolyData* vtkCalibratedModelRenderingPipeline::GetTrackingModel() const
{
  return m_TrackingModelTransformFilter->GetOutput();
}


//-----------------------------------------------------------------------------
vtkRenderWindow* vtkCalibratedModelRenderingPipeline::GetRenderWindow() const
{
  return m_RenderWin;
}


//-----------------------------------------------------------------------------
void vtkCalibratedModelRenderingPipeline::Render()
{
  m_Camera->Modified();
  m_TrackingModelActor->Modified();
  m_VisualisationModelActor->Modified();
  m_Renderer->Modified();

  m_RenderWin->Render();
}


//-----------------------------------------------------------------------------
void vtkCalibratedModelRenderingPipeline::DumpScreen(const std::string fileName)
{
  // Keep these local, or else the vtkWindowToImageFilter always appeared to cache its output,
  // regardless of the value of ShouldRerenderOn.

  vtkSmartPointer<vtkWindowToImageFilter> renderWindowToImageFilter = vtkSmartPointer<vtkWindowToImageFilter>::New();
  renderWindowToImageFilter->SetInput(m_RenderWin);
  renderWindowToImageFilter->SetInputBufferTypeToRGB();
  renderWindowToImageFilter->SetMagnification(1);
  renderWindowToImageFilter->ShouldRerenderOn();

  vtkSmartPointer<vtkPNGWriter> renderedImageWriter = vtkSmartPointer<vtkPNGWriter>::New();
  renderedImageWriter->SetInputConnection(renderWindowToImageFilter->GetOutputPort());
  renderedImageWriter->SetFileName(fileName.c_str());
  renderedImageWriter->Write();
}


//-----------------------------------------------------------------------------
vtkSmartPointer<vtkMatrix4x4> vtkCalibratedModelRenderingPipeline::GetTransform(const std::vector<float> &transform)
{
  double degreesToRadians = CV_PI/180.0;
  cv::Matx44d matCV = mitk::ConstructRigidTransformationMatrix(transform[0]*degreesToRadians, transform[1]*degreesToRadians, transform[2]*degreesToRadians, transform[3], transform[4], transform[5]);
  vtkSmartPointer<vtkMatrix4x4> mat = vtkSmartPointer<vtkMatrix4x4>::New();
  mitk::CopyToVTK4x4Matrix(matCV, *mat);
  return mat;
}


//-----------------------------------------------------------------------------
void vtkCalibratedModelRenderingPipeline::SetModelToWorldTransform(const std::vector<float> &transform)
{
  vtkSmartPointer<vtkMatrix4x4> mat = this->GetTransform(transform);
  this->SetModelToWorldMatrix(*mat);
}


//-----------------------------------------------------------------------------
void vtkCalibratedModelRenderingPipeline::SetModelToWorldMatrix(const vtkMatrix4x4& modelToWorld)
{
  m_ModelToWorldMatrix->DeepCopy(&modelToWorld);

  m_TrackingModelReader->Modified();
  m_TrackingModelTransformFilter->Modified();
  m_VisualisationModelReader->Modified();
  m_VisualisationModelTransformFilter->Modified();
}


//-----------------------------------------------------------------------------
void vtkCalibratedModelRenderingPipeline::SetCameraToWorldTransform(const std::vector<float> &transform)
{
  vtkSmartPointer<vtkMatrix4x4> mat = this->GetTransform(transform);
  this->SetCameraToWorldMatrix(*mat);
}


//-----------------------------------------------------------------------------
void vtkCalibratedModelRenderingPipeline::SetCameraToWorldMatrix(const vtkMatrix4x4& cameraToWorld)
{
  m_CameraToWorldMatrix->DeepCopy(&cameraToWorld);
  this->UpdateCamera();
}


//-----------------------------------------------------------------------------
void vtkCalibratedModelRenderingPipeline::SetWorldToCameraMatrix(const vtkMatrix4x4& worldToCamera)
{
  vtkSmartPointer<vtkMatrix4x4> mat = vtkSmartPointer<vtkMatrix4x4>::New();
  mat->DeepCopy(&worldToCamera);
  mat->Invert();
  this->SetCameraToWorldMatrix(*mat);
}


//-----------------------------------------------------------------------------
void vtkCalibratedModelRenderingPipeline::SetWorldToCameraTransform(const std::vector<float>& transform)
{
  vtkSmartPointer<vtkMatrix4x4> mat = this->GetTransform(transform);
  this->SetWorldToCameraMatrix(*mat);
}


//-----------------------------------------------------------------------------
void vtkCalibratedModelRenderingPipeline::SetIsRightHandCamera(const bool& isRight)
{
  m_IsRightHandCamera = isRight;
  this->UpdateCamera();
}


//-----------------------------------------------------------------------------
void vtkCalibratedModelRenderingPipeline::UpdateCamera()
{
  double origin[4]     = {0, 0,    0,    1};
  double focalPoint[4] = {0, 0,   1000, 1};
  double viewUp[4]     = {0, -1000, 0,    1};

  m_CameraMatrix->Identity();

  // Right-to-left is optional. It will also be an Identity matrix if it wasnt specified.
  if (m_IsRightHandCamera)
  {
    m_Camera->SetIntrinsicParameters(m_RightIntrinsicMatrix.at<double>(0,0), m_RightIntrinsicMatrix.at<double>(1,1), m_RightIntrinsicMatrix.at<double>(0,2), m_RightIntrinsicMatrix.at<double>(1,2));
    vtkMatrix4x4::Multiply4x4(m_CameraToWorldMatrix, m_RightToLeftMatrix, m_CameraMatrix);
  }
  else
  {
    m_Camera->SetIntrinsicParameters(m_LeftIntrinsicMatrix.at<double>(0,0), m_LeftIntrinsicMatrix.at<double>(1,1), m_LeftIntrinsicMatrix.at<double>(0,2), m_LeftIntrinsicMatrix.at<double>(1,2));
    m_CameraMatrix->DeepCopy(m_CameraToWorldMatrix);
  }

  m_CameraMatrix->MultiplyPoint(origin, origin);
  m_CameraMatrix->MultiplyPoint(focalPoint, focalPoint);
  m_CameraMatrix->MultiplyPoint(viewUp, viewUp);

  viewUp[0] = viewUp[0] - origin[0];
  viewUp[1] = viewUp[1] - origin[1];
  viewUp[2] = viewUp[2] - origin[2];

  m_Camera->SetPosition(origin[0], origin[1], origin[2]);
  m_Camera->SetFocalPoint(focalPoint[0], focalPoint[1], focalPoint[2]);
  m_Camera->SetViewUp(viewUp[0], viewUp[1], viewUp[2]);
  m_Camera->SetClippingRange(1, 10000);
  m_Camera->Modified();

  m_CameraMatrixInverted->DeepCopy(m_CameraMatrix);
  m_CameraMatrixInverted->Invert();
}


//-----------------------------------------------------------------------------
bool vtkCalibratedModelRenderingPipeline::IsFacingCamera(const double normal[3])
{
  bool result = false;

  double directionOfProjection[3];
  m_Camera->GetDirectionOfProjection(directionOfProjection);

  double projectionNormalised[3];
  niftk::NormaliseToUnitLength(directionOfProjection, projectionNormalised);

  projectionNormalised[0] *= -1;
  projectionNormalised[1] *= -1;
  projectionNormalised[2] *= -1;

  double normalNormalised[3];
  niftk::NormaliseToUnitLength(normal, normalNormalised); // just in case

  double angleInDegrees = niftk::AngleBetweenTwoUnitVectorsInDegrees(projectionNormalised, normalNormalised);

  if (fabs(angleInDegrees) < 80)
  {
    result = true;
  }
  return result;
}


//-----------------------------------------------------------------------------
void vtkCalibratedModelRenderingPipeline::ProjectToCameraSpace(const double worldPoint[3], double cameraPoint[3])
{
  double homogeneousWorldPoint[4];
  homogeneousWorldPoint[0] = worldPoint[0];
  homogeneousWorldPoint[1] = worldPoint[1];
  homogeneousWorldPoint[2] = worldPoint[2];
  homogeneousWorldPoint[3] = 1;

  double homogeneousCameraPoint[4];

  m_CameraMatrixInverted->MultiplyPoint(homogeneousWorldPoint, homogeneousCameraPoint);

  cameraPoint[0] = homogeneousCameraPoint[0];
  cameraPoint[1] = homogeneousCameraPoint[1];
  cameraPoint[2] = homogeneousCameraPoint[2];
}


//-----------------------------------------------------------------------------
void vtkCalibratedModelRenderingPipeline::ProjectPoint(const double worldPoint[3], double imagePoint[2])
{
  double cameraPoint[3];
  this->ProjectToCameraSpace(worldPoint, cameraPoint);

  CvMat *worldPointCV = cvCreateMat(1,3,CV_64FC1);
  CvMat *cameraRotationVector = cvCreateMat(1,3,CV_64FC1);
  CvMat *cameraTranslationVector = cvCreateMat(1,3,CV_64FC1);
  CvMat *imagePointCV = cvCreateMat(1,2,CV_64FC1);
  CvMat *distortion = cvCreateMat(1,4,CV_64FC1);
  CvMat *intrinsics = cvCreateMat(3,3,CV_64FC1);

  for (int i = 0; i < 3; i++)
  {
    CV_MAT_ELEM(*cameraRotationVector, double, 0, i) = 0; // not needed as we used VTK matrix
    CV_MAT_ELEM(*cameraTranslationVector, double, 0, i) = 0; // not needed as we used VTK matrix
    CV_MAT_ELEM(*worldPointCV, double, 0, i) = cameraPoint[i];
  }
  if (m_UseDistortion)
  {
    if (m_IsRightHandCamera)
    {
      for (int i = 0; i < 4; i++)
      {
        CV_MAT_ELEM(*distortion, double, 0, i) = m_RightDistortionVector.at<double>(0,i);
      }
    }
    else
    {
      for (int i = 0; i < 4; i++)
      {
        CV_MAT_ELEM(*distortion, double, 0, i) = m_LeftDistortionVector.at<double>(0,i);
      }
    }
  }
  else
  {
    for (int i = 0; i < 4; i++)
    {
      CV_MAT_ELEM(*distortion, double, 0, i) = 0;
    }
  }
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      if (m_IsRightHandCamera)
      {
        CV_MAT_ELEM(*intrinsics, double, i, j) = m_RightIntrinsicMatrix.at<double>(i,j);
      }
      else
      {
        CV_MAT_ELEM(*intrinsics, double, i, j) = m_LeftIntrinsicMatrix.at<double>(i,j);
      }
    }
  }

  cvProjectPoints2(
      worldPointCV,
      cameraRotationVector,
      cameraTranslationVector,
      intrinsics,
      distortion,
      imagePointCV
      );


  imagePoint[0] = CV_MAT_ELEM(*imagePointCV, double, 0, 0);
  imagePoint[1] = CV_MAT_ELEM(*imagePointCV, double, 0, 1);

  // Basically equivalent.
  // imagePoint[0] = (cameraPoint[0]/cameraPoint[2])*m_LeftIntrinsicMatrix.at<double>(0,0) + m_LeftIntrinsicMatrix.at<double>(0,2);
  // imagePoint[1] = (cameraPoint[1]/cameraPoint[2])*m_LeftIntrinsicMatrix.at<double>(1,1) + m_LeftIntrinsicMatrix.at<double>(1,2);

  cvReleaseMat(&worldPointCV);
  cvReleaseMat(&cameraRotationVector);
  cvReleaseMat(&cameraTranslationVector);
  cvReleaseMat(&imagePointCV);
  cvReleaseMat(&distortion);
  cvReleaseMat(&intrinsics);
}

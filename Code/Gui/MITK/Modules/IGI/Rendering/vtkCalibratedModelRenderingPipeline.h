/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef vtkCalibratedModelRenderingPipeline_h
#define vtkCalibratedModelRenderingPipeline_h

#include "niftkIGIExports.h"

#include <cv.h>

#include <mitkVector.h>
#include <mitkPoint.h>

#include <vtkObject.h>
#include <vtkIndent.h>
#include <vtkSmartPointer.h>
#include <vtkOpenGLMatrixDrivenCamera.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkPNGReader.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkMatrixToLinearTransform.h>
#include <vtkSphereSource.h>
#include <vtkGlyph3D.h>
#include <vtkWindowToImageFilter.h>
#include <vtkPNGWriter.h>
#include <vtkBMPWriter.h>
#include <vtkSetGet.h>
#include <vtkPlaneSource.h>
#include <vtkTexture.h>
#include <vtkImageFlip.h>

/**
 * \class vtkCalibratedModelRenderingPipeline
 * \brief Used to quickly make test harnesses demos and research prototypes.
 *
 * Loads camera calibration files and displays a perspective projection
 * without doing distortion correction (i.e. just does intrinsic and extrinsic).
 *
 */
class NIFTKIGI_EXPORT vtkCalibratedModelRenderingPipeline {

public:

  /**
   * @brief Constructor.
   * @param name
   * @param windowSize The actual window size we want to represent (e.g. 1920, 1080)
   * @param calibratedWindowSize For calibrated camera model, we specify the size of the window that the calibration (e.g. niftkCameraCalibration) was done at. (e.g. 1920,540)
   * @param leftIntrinsicsFileName Filename containing intrinsic parameters, as output by other NifTK programs such as niftkCameraCalibration.
   * @param rightIntrinsicsFileName Filename containing intrinsic parameters, as output by other NifTK programs such as niftkCameraCalibration.
   * @param visualisationModelFileName Filename of vtkPolyData that will be rendered at a given pose.
   * @param textureFileName If not-empty, and if loadable, will be texture mapped onto the visualisationModel.
   * @param rightToLeftFileName If not-empty, and if loadable, will be added into camera matrices to simulate a right-hand camera. Default is left.
   * @param trackingModelFileName If not-empty, and if loadable, will also be rendered, where each surface point is combined with a sphere glyph.
   * @param  ultrasoundCalibrationMatrixFileName If not-empty, and if loadable, and if ultrasoundImageFileName is also set, will also render the ultrasound image in place.
   * @param  ultrasoundImageFileName If not-empty, and if loadable, and if ultrasoundCalibrationMatrixFileName is also set, will also render the ultrasound image in place.
   */
  vtkCalibratedModelRenderingPipeline(
    const std::string& name,
    const mitk::Point2I& windowSize,
    const mitk::Point2I& calibratedWindowSize,
    const std::string& leftIntrinsicsFileName,
    const std::string& rightIntrinsicsFileName,
    const std::string& visualisationModelFileName,
    const std::string& rightToLeftFileName,
    const std::string& textureFileName,
    const std::string& trackingModelFileName,
    const std::string& ultrasoundCalibrationMatrixFileName,
    const std::string& ultrasoundImageFileName,
    const float& trackingGlyphRadius
    );

  virtual ~vtkCalibratedModelRenderingPipeline();

  /**
   * @brief Returns the name of this object, just used for reference.
   */
  std::string GetName() const { return m_Name; }

  /**
   * @brief This class can be setup to be a right or left handed camera in a stereo camera pair.
   */
  void SetIsRightHandCamera(const bool& isRight);

  /**
   * @brief Get the right(true)/left(false) indicator.
   */
  bool GetIsRightHandCamera() const { return m_IsRightHandCamera; }

  /**
   * @brief Updates the whole VTK pipeline.
   */
  virtual void Render();

  /**
   * @brief Updates screen and writes to specified file.
   * @param fileName file name
   */
  void DumpScreen(const std::string fileName);

  /**
   * @brief Saves the Model to World transform to file.
   * @param fileName
   */
  void SaveModelToWorld(const std::string& fileName);

  /**
   * @brief Saves the Camera to World transform to file.
   * @param fileName
   */
  void SaveCameraToWorld(const std::string& fileName);

  /**
   * @brief Multiplies model vtkPolyData points by modelToWorld to give a position in world coordinates,
   * having the effect of moving the model relative to the world coordinates.
   * @param modelToWorld rigid body transform
   */
  void SetModelToWorldMatrix(const vtkMatrix4x4& modelToWorld);

  /**
   * @brief Sets transform using rx,ry,rz (degrees), tx,ty,rz (mm).
   */
  void SetModelToWorldTransform(const std::vector<float>&);

  /**
   * @brief Used to set the camera position in world coordinates.
   * @param cameraToWorld rigid body transform
   */
  void SetCameraToWorldMatrix(const vtkMatrix4x4& cameraToWorld);

  /**
   * @brief Sets transform using rx,ry,rz (degrees), tx,ty,rz (mm).
   */
  void SetCameraToWorldTransform(const std::vector<float>&);

  /**
   * @brief Used to set the camera position in world coordinates.
   * @param worldToCamera rigid body transform
   */
  void SetWorldToCameraMatrix(const vtkMatrix4x4& worldToCamera);

  /**
   * @brief Sets transform using rx,ry,rz (degrees), tx,ty,rz (mm).
   */
  void SetWorldToCameraTransform(const std::vector<float>&);

  /**
   * @brief Returns true if point is deemed to be facing camera.
   * @normal normal vector of a point in world coordinates
   */
  bool IsFacingCamera(const double normal[3]);

  /**
   * @brief projects a point from world 3D to image 2D.
   */
  void ProjectPoint(const double world[3], double imagePoint[2]);

  /**
   * @brief projects a point from world 3D to camera 3D.
   */
  void ProjectToCameraSpace(const double worldPoint[3], double cameraPoint[3]);

  /**
   * @brief Returns a pointer to the tracking model, so external classes can iterate over points for instance.
   */
  vtkPolyData* GetTrackingModel() const;

  /**
   * @brief Returns a point to the vtkRenderWindow.
   */
  vtkRenderWindow* GetRenderWindow() const;

private:

  vtkCalibratedModelRenderingPipeline(const vtkCalibratedModelRenderingPipeline&);  // Purposefully not implemented.
  void operator=(const vtkCalibratedModelRenderingPipeline&);  // Purposefully not implemented.

  void UpdateCamera();
  void UpdateUltrasoundPlanePosition();
  vtkSmartPointer<vtkMatrix4x4> GetTransform(const std::vector<float> &transform);

  std::string                                  m_Name;

  bool                                         m_UseDistortion;
  bool                                         m_IsRightHandCamera;
  cv::Mat                                      m_LeftIntrinsicMatrix;
  cv::Mat                                      m_LeftDistortionVector;
  cv::Mat                                      m_RightIntrinsicMatrix;
  cv::Mat                                      m_RightDistortionVector;

  vtkSmartPointer<vtkOpenGLMatrixDrivenCamera> m_Camera;
  vtkSmartPointer<vtkMatrix4x4>                m_RightToLeftMatrix;

  vtkSmartPointer<vtkMatrix4x4>                m_ModelToWorldMatrix;
  vtkSmartPointer<vtkMatrixToLinearTransform>  m_ModelToWorldTransform;
  vtkSmartPointer<vtkMatrix4x4>                m_CameraToWorldMatrix;
  vtkSmartPointer<vtkMatrix4x4>                m_CameraMatrix; // combines m_CameraToWorldMatrix and m_RightToLeftMatrix if necessary.
  vtkSmartPointer<vtkMatrix4x4>                m_CameraMatrixInverted;

  vtkSmartPointer<vtkPolyDataReader>           m_TrackingModelReader;
  vtkSmartPointer<vtkSphereSource>             m_SphereForGlyph;
  vtkSmartPointer<vtkGlyph3D>                  m_GlyphFilter;
  vtkSmartPointer<vtkTransformPolyDataFilter>  m_TrackingModelTransformFilter;
  vtkSmartPointer<vtkPolyDataMapper>           m_TrackingModelMapper;
  vtkSmartPointer<vtkActor>                    m_TrackingModelActor;
  vtkSmartPointer<vtkPolyDataWriter>           m_TrackingModelWriter;

  vtkSmartPointer<vtkPNGReader>                m_TextureReader;
  vtkSmartPointer<vtkTexture>                  m_Texture;
  vtkSmartPointer<vtkPolyDataReader>           m_VisualisationModelReader;
  vtkSmartPointer<vtkTransformPolyDataFilter>  m_VisualisationModelTransformFilter;
  vtkSmartPointer<vtkPolyDataMapper>           m_VisualisationModelMapper;
  vtkSmartPointer<vtkActor>                    m_VisualisationModelActor;
  vtkSmartPointer<vtkPolyDataWriter>           m_VisualisationModelWriter;

  vtkSmartPointer<vtkMatrix4x4>                m_UltrasoundCalibrationMatrix;
  vtkSmartPointer<vtkMatrix4x4>                m_UltrasoundTransformMatrix;
  vtkSmartPointer<vtkPNGReader>                m_UltrasoundImageReader;
  vtkSmartPointer<vtkImageFlip>                m_UltrasoundImageYAxisFlipper;
  vtkSmartPointer<vtkPlaneSource>              m_UltrasoundImagePlane;
  vtkSmartPointer<vtkTexture>                  m_UltrasoundImageTexture;
  vtkSmartPointer<vtkPolyDataMapper>           m_UltrasoundImageMapper;
  vtkSmartPointer<vtkActor>                    m_UltrasoundImageActor;

  vtkSmartPointer<vtkRenderer>                 m_Renderer;
  vtkSmartPointer<vtkRenderWindow>             m_RenderWin;
};

#endif

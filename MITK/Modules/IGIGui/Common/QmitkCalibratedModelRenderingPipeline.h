/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkCalibratedModelRenderingPipeline_h
#define QmitkCalibratedModelRenderingPipeline_h

#include "niftkIGIGuiExports.h"
#include <QVTKWidget.h>
#include <QKeyEvent>
#include <QEvent>
#include <QString>
#include <vtkCalibratedModelRenderingPipeline.h>

/**
 * \class QmitkCalibratedModelRenderingPipeline
 * \brief Harness to call vtkCalibratedModelRenderingPipeline.
 */
class NIFTKIGIGUI_EXPORT QmitkCalibratedModelRenderingPipeline : public QVTKWidget
{
  Q_OBJECT

public:

  QmitkCalibratedModelRenderingPipeline(
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
    const float& trackingGlyphRadius,
    const std::string& outputData,
    QWidget *parent = 0
  );

  virtual ~QmitkCalibratedModelRenderingPipeline();

  /**
   * @see vtkCalibratedModelRenderingPipeline::SetModelToWorldMatrix()
   */
  void SetModelToWorldMatrix(const vtkMatrix4x4& modelToWorld);

  /**
   * @see vtkCalibratedModelRenderingPipeline::SetModelToWorldTransform()
   */
  void SetModelToWorldTransform(const std::vector<float>&);

  /**
   * @see vtkCalibratedModelRenderingPipeline::SetCameraToWorldMatrix()
   */
  void SetCameraToWorldMatrix(const vtkMatrix4x4& cameraToWorld);

  /**
   * @see vtkCalibratedModelRenderingPipeline::SetCameraToWorldTransform()
   */
  void SetCameraToWorldTransform(const std::vector<float>&);

  /**
   * @see vtkCalibratedModelRenderingPipeline::SetWorldToCameraMatrix()
   */
  void SetWorldToCameraMatrix(const vtkMatrix4x4& worldToCamera);

  /**
   * @see vtkCalibratedModelRenderingPipeline::SetWorldToCameraTransform()
   */
  void SetWorldToCameraTransform(const std::vector<float>&);

  /**
   * @see vtkCalibratedModelRenderingPipeline::SetUseRightToLeft()
   */
  void SetIsRightHandCamera(const bool& isRight);

  /**
   * @see vtkCalibratedModelRenderingPipeline::Render()
   */
  void Render();

  /**
   * @brief Takes a screenshot from the perspective of the left camera, outputing to <outputData>.left.png
   */
  void SaveLeftImage();

  /**
   * @brief Takes a screenshot from the perspective of the right camera, outputing to <outputData>.right.png
   */
  void SaveRightImage();

  /**
   * @brief Calls SaveLeftImage(), then SaveRightImage(), then dump 3D points and projected 2D points to <outputData>
   */
  void SaveData();

private:

  vtkCalibratedModelRenderingPipeline m_Pipeline;
  mitk::Point2I m_WindowSize;
  mitk::Point2I m_CalibratedWindowSize;
  std::string m_OutputData;

};

#endif

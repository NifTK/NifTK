/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkSingle3DView_h
#define QmitkSingle3DView_h

#include "niftkCoreGuiExports.h"
#include <mitkRenderWindowFrame.h>
#include <mitkGradientBackground.h>
#include <mitkDataStorage.h>
#include "QmitkCmicLogo.h"
#include "QmitkBitmapOverlay.h"
#include <QWidget>
#include <QFrame>
#include <QResizeEvent>
#include <QmitkRenderWindow.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>

class QGridLayout;
class QmitkRenderWindow;

namespace mitk {
  class RenderingManager;
}

/**
 * \class QmitkSingle3DView
 * \brief Widget containing a single 3D render window whose purpose
 * is to render a 3D VTK scene, blended with RGB image data, such as
 * might be available from video or ultrasound.
 *
 * This class was originally a stripped down QmitkStdMultiWidget, but
 * now it has been tidied up, it really is no longer a QmitkStdMultiWidget.
 */
class NIFTKCOREGUI_EXPORT QmitkSingle3DView : public QWidget
{
  Q_OBJECT

public:

  QmitkSingle3DView(QWidget* parent = 0, Qt::WindowFlags f = 0, mitk::RenderingManager* renderingManager = 0);
  virtual ~QmitkSingle3DView();

  /**
   * \brief Stores ds locally, and sets the data storage on the contained
   * QmitkBitmapOverlay and QmitkRenderWindow.
   */
  void SetDataStorage( mitk::DataStorage* ds );

  /**
   * \brief Returns a pointer to the contained QmitkRenderWindow.
   */
  QmitkRenderWindow* GetRenderWindow() const;

  /**
   * \brief Retrieves the opacity from the QmitkBitmapOverlay.
   */
  float GetOpacity() const;

  /**
   * \brief Sets the opacity on the QmitkBitmapOverlay.
   * \param value [0..1]
   */
  void SetOpacity(const float& value);

  /**
   * \brief Passes the node onto the QmitkBitmapOverlay, so that
   * the QmitkBitmapOverlay can use it for a background or foreground image.
   * \param node an mitk::DataNode that should contain an RGB image.
   */
  void SetImageNode(const mitk::DataNode* node);

  /**
   * \brief Sets a node, which should contain an mitk::CoordinateAxesData,
   * which is used for moving the view-point around.
   *
   * For example, if this viewer is to render like a tracked laparoscope,
   * then this transform node should be the transform node from the tracker attached
   * to the laparoscope, and this class will then render a laparoscope viewpoint.
   */
  void SetTransformNode(const mitk::DataNode* node);

  /**
   * \brief Calls mitk::GradientBackground::EnableGradientBackground().
   */
  void EnableGradientBackground();

  /**
   * \brief Calls mitk::GradientBackground::DisableGradientBackground().
   */
  void DisableGradientBackground();

  /**
   * \brief Calls mitk::GradientBackground::SetGradientColors().
   */
  void SetGradientBackgroundColors( const mitk::Color & upper, const mitk::Color & lower );

  /**
   * \brief Calls mitk::CMICLogo::EnableDepartmentLogo(), and is currently unused.
   */
  void EnableDepartmentLogo();

  /**
   * \brief Calls mitk::CMICLogo::DisableDepartmentLogo(), and is currently unused.
   */
  void DisableDepartmentLogo();

  /**
   * \brief Calls mitk::CMICLogo::SetDepartmentLogoPath(), and is currently unused.
   */
  void SetDepartmentLogoPath( const char * path );

  /**
   * \brief Sets the Calibration file name, which causes a re-loading of the calibration matrix.
   */
  void SetCalibrationFileName(const std::string& fileName);

  /**
   * \brief Method responsible for making sure the Display Geometry can view
   * the currently visible data in the DataStorage.
   *
   * This is not the method used to adjust the camera position for the QmitkBitmapOverlay.
   */
  void Fit();

protected:

  /**
   * \brief Re-implemented so we can tell QmitkBitmapOverlay the display size has changed.
   */
  virtual void resizeEvent(QResizeEvent* event);

  mitk::DataStorage::Pointer         m_DataStorage;
  QmitkRenderWindow                 *m_RenderWindow;
  QGridLayout                       *m_Layout;
  mitk::RenderingManager            *m_RenderingManager;
  mitk::RenderWindowFrame::Pointer   m_RenderWindowFrame;
  mitk::GradientBackground::Pointer  m_GradientBackground;
  CMICLogo::Pointer                  m_LogoRendering;
  QmitkBitmapOverlay::Pointer        m_BitmapOverlay;
  std::string                        m_CalibrationFileName;
  vtkSmartPointer<vtkMatrix4x4>      m_CalibrationTransform;

};
#endif /* QmitkSingle3DView */

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkSingle3DViewWidget_h
#define niftkSingle3DViewWidget_h

#include "niftkIGIOverlayEditorExports.h"
#include <QmitkCmicLogo.h>

#include <mitkRenderWindowFrame.h>
#include <mitkGradientBackground.h>
#include <mitkDataStorage.h>
#include <mitkImage.h>
#include <QmitkRenderWindow.h>
#include <QWidget>
#include <QFrame>
#include <QResizeEvent>

class QGridLayout;
class QmitkRenderWindow;

namespace mitk {
  class RenderingManager;
}

namespace niftk
{

/**
 * \class Single3DViewWidget
 * \brief Base class widget containing a single 3D render window whose purpose
 * is to render a 3D VTK scene, blended with image data, such as
 * might be available from video or ultrasound.
 */
class NIFTKIGIOVERLAYEDITOR_EXPORT Single3DViewWidget : public QWidget
{
  Q_OBJECT

public:

  Single3DViewWidget(QWidget* parent = 0,
                     Qt::WindowFlags f = 0,
                     mitk::RenderingManager* renderingManager = 0);

  virtual ~Single3DViewWidget();

  /**
   * \brief Returns a pointer to the contained QmitkRenderWindow.
   */
  QmitkRenderWindow* GetRenderWindow() const;

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
  void SetGradientBackgroundColors(const mitk::Color& upper, const mitk::Color& lower);

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
  void SetDepartmentLogoPath(const QString& path);

  /**
   * \brief Stores ds locally, and sets the data storage on the contained
   * niftk::BitmapOverlay and QmitkRenderWindow.
   */
  virtual void SetDataStorage(mitk::DataStorage* ds);

  /**
   * \brief Stores the node locally.
   */
  virtual void SetImageNode(const mitk::DataNode* node);

  void SetClippingRange(const double& near, const double& far);

protected:

  /**
   * \brief Re-implemented so we can correctly scale image while resizeing.
   */
  virtual void resizeEvent(QResizeEvent* event) override;

  /**
   * \brief Derived classes need to update this to correclty rescale.
   */
  virtual void Update() = 0;

  /**
   * \brief Called when a DataStorage Node Removed Event was emitted.
   */
  virtual void NodeRemoved(const mitk::DataNode* node) {}

  /**
   * \brief Called when a DataStorage Node Changes Event was emitted.
   */
  virtual void NodeChanged(const mitk::DataNode* node) {}

  /**
   * \brief Called when a DataStorage Node Added Event was emitted.
   */
  virtual void NodeAdded(const mitk::DataNode* node) {}

  mitk::DataStorage::Pointer                    m_DataStorage;
  mitk::Image::Pointer                          m_Image;
  mitk::DataNode::Pointer                       m_ImageNode;
  double                                        m_ClippingRange[2];

private:

  void DeRegisterDataStorageListeners();
  void InternalNodeRemoved(const mitk::DataNode* node);
  void InternalNodeChanged(const mitk::DataNode* node);
  void InternalNodeAdded(const mitk::DataNode* node);

  mitk::RenderingManager                       *m_RenderingManager;
  QmitkRenderWindow                            *m_RenderWindow;
  QGridLayout                                  *m_Layout;
  mitk::RenderWindowFrame::Pointer              m_RenderWindowFrame;
  mitk::GradientBackground::Pointer             m_GradientBackground;
  CMICLogo::Pointer                             m_LogoRendering;
};

} // end namespace

#endif

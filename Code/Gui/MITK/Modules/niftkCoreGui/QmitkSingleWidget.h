/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QmitkSingleWidget_h
#define QmitkSingleWidget_h

#include "niftkCoreGuiExports.h"
#include <mitkRenderWindowFrame.h>
#include <mitkGradientBackground.h>
#include <mitkDataStorage.h>
#include "QmitkCmicLogo.h"
#include "QmitkBitmapOverlay.h"
#include <QWidget>
#include <QFrame>
#include <QmitkRenderWindow.h>

class QGridLayout;
class QmitkRenderWindow;

namespace mitk {
  class RenderingManager;
}

/**
 * \class QmitkSingleWidget
 * \brief Widget containing a 3D render window, and to overlay data ontop of an RGB image.
 */
class NIFTKCOREGUI_EXPORT QmitkSingleWidget : public QWidget
{
  Q_OBJECT

public:

  QmitkSingleWidget(QWidget* parent = 0, Qt::WindowFlags f = 0, mitk::RenderingManager* renderingManager = 0);
  virtual ~QmitkSingleWidget();

  void SetDataStorage( mitk::DataStorage* ds );

  QmitkRenderWindow* GetRenderWindow() const;

  float GetOpacity() const;
  void SetOpacity(const float& value);

  void SetImageNode(const mitk::DataNode* node);
  void SetTransformNode(const mitk::DataNode* node);

  void EnableGradientBackground();
  void DisableGradientBackground();
  void SetGradientBackgroundColors( const mitk::Color & upper, const mitk::Color & lower );

  void EnableDepartmentLogo();
  void DisableDepartmentLogo();
  void SetDepartmentLogoPath( const char * path );

  void Fit();

protected:

  mitk::DataStorage::Pointer         m_DataStorage;
  QmitkRenderWindow                 *m_RenderWindow;
  QGridLayout                       *m_Layout;
  mitk::RenderingManager            *m_RenderingManager;
  mitk::RenderWindowFrame::Pointer   m_RenderWindowFrame;
  mitk::GradientBackground::Pointer  m_GradientBackground;
  CMICLogo::Pointer                  m_LogoRendering;
  QmitkBitmapOverlay::Pointer        m_BitmapOverlay;

};
#endif /* QmitkSingleWidget */

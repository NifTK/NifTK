/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkSingleWidget.h"
#include <QGridLayout>

//-----------------------------------------------------------------------------
QmitkSingleWidget::QmitkSingleWidget(QWidget* parent, Qt::WindowFlags f, mitk::RenderingManager* renderingManager)
: QWidget(parent, f)
, m_DataStorage(NULL)
, m_RenderWindow(NULL)
, m_Layout(NULL)
, m_RenderingManager(NULL)
, m_RenderWindowFrame(NULL)
, m_GradientBackground(NULL)
, m_LogoRendering(NULL)
, m_BitmapOverlay(NULL)
{
  /******************************************************
   * Use the global RenderingManager if none was specified
   ******************************************************/
  if (m_RenderingManager == NULL)
  {
    m_RenderingManager = mitk::RenderingManager::GetInstance();
  }

  m_RenderWindow = new QmitkRenderWindow(NULL, "single.widget1", NULL, m_RenderingManager);
  m_RenderWindow->setMaximumSize(2000,2000);
  m_RenderWindow->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  m_RenderWindow->GetRenderer()->GetVtkRenderer()->InteractiveOff();

  m_Layout = new QGridLayout(this);
  m_Layout->setContentsMargins(0, 0, 0, 0);
  m_Layout->addWidget(m_RenderWindow);

  mitk::BaseRenderer::GetInstance(m_RenderWindow->GetRenderWindow())->SetMapperID(mitk::BaseRenderer::Standard3D);

  m_GradientBackground = mitk::GradientBackground::New();
  m_GradientBackground->SetRenderWindow(m_RenderWindow->GetRenderWindow());
  m_GradientBackground->SetGradientColors(0, 0, 0, 0, 0, 0);
  m_GradientBackground->Enable();

  m_LogoRendering = CMICLogo::New();
  m_LogoRendering->SetRenderWindow(m_RenderWindow->GetRenderWindow());
  m_LogoRendering->Disable();

  m_BitmapOverlay = QmitkBitmapOverlay::New();
  m_BitmapOverlay->SetRenderWindow(m_RenderWindow->GetRenderWindow());

  m_RenderWindowFrame = mitk::RenderWindowFrame::New();
  m_RenderWindowFrame->SetRenderWindow(m_RenderWindow->GetRenderWindow());
  m_RenderWindowFrame->Enable(1.0,0.0,0.0);
}


//-----------------------------------------------------------------------------
QmitkSingleWidget::~QmitkSingleWidget()
{
}


//-----------------------------------------------------------------------------
void QmitkSingleWidget::SetDataStorage( mitk::DataStorage* ds )
{
  m_DataStorage = ds;
  mitk::BaseRenderer::GetInstance(m_RenderWindow->GetRenderWindow())->SetDataStorage(ds);

  if (m_DataStorage.IsNotNull())
  {
    m_BitmapOverlay->SetDataStorage (m_DataStorage);
    m_BitmapOverlay->Enable();
  }
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* QmitkSingleWidget::GetRenderWindow() const
{
  return m_RenderWindow;
}


//-----------------------------------------------------------------------------
float QmitkSingleWidget::GetOpacity() const
{
  return static_cast<float>(m_BitmapOverlay->GetOpacity());
}


//-----------------------------------------------------------------------------
void QmitkSingleWidget::SetOpacity(const float& value)
{
  m_BitmapOverlay->SetOpacity(value);
}


//-----------------------------------------------------------------------------
void QmitkSingleWidget::SetImageNode(const mitk::DataNode* node)
{
  m_BitmapOverlay->SetNode(node);
}


//-----------------------------------------------------------------------------
void QmitkSingleWidget::SetTransformNode(const mitk::DataNode* node)
{
}


//-----------------------------------------------------------------------------
void QmitkSingleWidget::EnableGradientBackground()
{
  m_GradientBackground->Enable();
}


//-----------------------------------------------------------------------------
void QmitkSingleWidget::DisableGradientBackground()
{
  m_GradientBackground->Disable();
}


//-----------------------------------------------------------------------------
void QmitkSingleWidget::SetGradientBackgroundColors( const mitk::Color & upper, const mitk::Color & lower )
{
  m_GradientBackground->SetGradientColors(upper[0], upper[1], upper[2], lower[0], lower[1], lower[2]);
}


//-----------------------------------------------------------------------------
void QmitkSingleWidget::EnableDepartmentLogo()
{
   m_LogoRendering->Enable();
}


//-----------------------------------------------------------------------------
void QmitkSingleWidget::DisableDepartmentLogo()
{
   m_LogoRendering->Disable();
}


//-----------------------------------------------------------------------------
void QmitkSingleWidget::SetDepartmentLogoPath( const char * path )
{
  m_LogoRendering->SetLogoSource(path);
}


//-----------------------------------------------------------------------------
void QmitkSingleWidget::Fit()
{
  mitk::BaseRenderer::GetInstance(m_RenderWindow->GetRenderWindow())->GetDisplayGeometry()->Fit();

  int w = vtkObject::GetGlobalWarningDisplay();
  vtkObject::GlobalWarningDisplayOff();

  vtkRenderer *vtkRenderer = mitk::BaseRenderer::GetInstance(m_RenderWindow->GetRenderWindow())->GetVtkRenderer();
  if ( vtkRenderer!= NULL )
  {
    vtkRenderer->ResetCamera();
  }

  vtkObject::SetGlobalWarningDisplay(w);
}




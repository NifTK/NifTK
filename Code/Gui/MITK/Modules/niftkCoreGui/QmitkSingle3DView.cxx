/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkSingle3DView.h"
#include <QGridLayout>

//-----------------------------------------------------------------------------
QmitkSingle3DView::QmitkSingle3DView(QWidget* parent, Qt::WindowFlags f, mitk::RenderingManager* renderingManager)
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
QmitkSingle3DView::~QmitkSingle3DView()
{
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::SetDataStorage( mitk::DataStorage* ds )
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
QmitkRenderWindow* QmitkSingle3DView::GetRenderWindow() const
{
  return m_RenderWindow;
}


//-----------------------------------------------------------------------------
float QmitkSingle3DView::GetOpacity() const
{
  return static_cast<float>(m_BitmapOverlay->GetOpacity());
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::SetOpacity(const float& value)
{
  m_BitmapOverlay->SetOpacity(value);
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::SetImageNode(const mitk::DataNode* node)
{
  m_BitmapOverlay->SetNode(node);
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::SetTransformNode(const mitk::DataNode* node)
{
  if (node != NULL)
  {
    // Todo: Do something sensible.
    std::cerr << "QmitkSingle3DView::SetTransformNode node=" << node << ", name=" << node->GetName() << std::endl;
  }
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::EnableGradientBackground()
{
  m_GradientBackground->Enable();
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::DisableGradientBackground()
{
  m_GradientBackground->Disable();
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::SetGradientBackgroundColors( const mitk::Color & upper, const mitk::Color & lower )
{
  m_GradientBackground->SetGradientColors(upper[0], upper[1], upper[2], lower[0], lower[1], lower[2]);
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::EnableDepartmentLogo()
{
   m_LogoRendering->Enable();
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::DisableDepartmentLogo()
{
   m_LogoRendering->Disable();
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::SetDepartmentLogoPath( const char * path )
{
  m_LogoRendering->SetLogoSource(path);
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::resizeEvent(QResizeEvent* /*event*/)
{
  m_BitmapOverlay->SetupCamera();
}


//-----------------------------------------------------------------------------
void QmitkSingle3DView::Fit()
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




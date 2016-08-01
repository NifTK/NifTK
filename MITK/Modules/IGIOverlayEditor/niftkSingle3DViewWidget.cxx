/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkSingle3DViewWidget.h"

#include <QGridLayout>
#include <mitkBaseGeometry.h>

namespace niftk
{
//-----------------------------------------------------------------------------
Single3DViewWidget::Single3DViewWidget(QWidget* parent,
                                       Qt::WindowFlags f,
                                       mitk::RenderingManager* renderingManager)
: QWidget(parent, f)
, m_DataStorage(nullptr)
, m_Image(nullptr)
, m_ImageNode(nullptr)
, m_RenderingManager(renderingManager)
, m_RenderWindow(nullptr)
, m_Layout(nullptr)
, m_RenderWindowFrame(nullptr)
, m_GradientBackground(nullptr)
, m_LogoRendering(nullptr)
{
  /******************************************************
   * Use the global RenderingManager if none was specified
   ******************************************************/
  if (m_RenderingManager == NULL)
  {
    m_RenderingManager = mitk::RenderingManager::GetInstance();
  }

  m_RenderWindow = new QmitkRenderWindow(this, "single.widget1", NULL, m_RenderingManager);
  m_RenderWindow->setMaximumSize(2000,2000);
  m_RenderWindow->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

  m_Layout = new QGridLayout(this);
  m_Layout->setContentsMargins(0, 0, 0, 0);
  m_Layout->addWidget(m_RenderWindow);

  mitk::BaseRenderer::GetInstance(m_RenderWindow->GetRenderWindow())->SetMapperID(mitk::BaseRenderer::Standard3D);

  m_RenderWindow->GetRenderer()->GetVtkRenderer()->InteractiveOff();
  m_RenderWindow->GetVtkRenderWindow()->GetInteractor()->Disable();

  m_GradientBackground = mitk::GradientBackground::New();
  m_GradientBackground->SetRenderWindow(m_RenderWindow->GetRenderWindow());
  m_GradientBackground->SetGradientColors(0, 0, 0, 0, 0, 0);
  m_GradientBackground->Enable();

  m_LogoRendering = CMICLogo::New();
  m_LogoRendering->SetRenderWindow(m_RenderWindow->GetRenderWindow());
  m_LogoRendering->Disable();

  m_RenderWindowFrame = mitk::RenderWindowFrame::New();
  m_RenderWindowFrame->SetRenderWindow(m_RenderWindow->GetRenderWindow());
  m_RenderWindowFrame->Enable(1.0,0.0,0.0);

  m_ClippingRange[0] = 2.0;
  m_ClippingRange[1] = 5000.0;
}


//-----------------------------------------------------------------------------
Single3DViewWidget::~Single3DViewWidget()
{
  this->DeRegisterDataStorageListeners();
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::DeRegisterDataStorageListeners()
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->RemoveNodeEvent.RemoveListener
      (mitk::MessageDelegate1<Single3DViewWidget, const mitk::DataNode*>
       (this, &Single3DViewWidget::InternalNodeRemoved ) );

    m_DataStorage->ChangedNodeEvent.RemoveListener
      (mitk::MessageDelegate1<Single3DViewWidget, const mitk::DataNode*>
      (this, &Single3DViewWidget::InternalNodeChanged ) );

    m_DataStorage->ChangedNodeEvent.RemoveListener
      (mitk::MessageDelegate1<Single3DViewWidget, const mitk::DataNode*>
      (this, &Single3DViewWidget::InternalNodeAdded ) );
  }
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::SetDataStorage( mitk::DataStorage* dataStorage )
{
  if (m_DataStorage.IsNotNull() && m_DataStorage != dataStorage)
  {
    this->DeRegisterDataStorageListeners();
  }

  m_DataStorage = dataStorage;

  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->RemoveNodeEvent.AddListener
      (mitk::MessageDelegate1<Single3DViewWidget, const mitk::DataNode*>
       (this, &Single3DViewWidget::InternalNodeRemoved ) );

    m_DataStorage->ChangedNodeEvent.AddListener
      (mitk::MessageDelegate1<Single3DViewWidget, const mitk::DataNode*>
      (this, &Single3DViewWidget::InternalNodeChanged ) );

    m_DataStorage->ChangedNodeEvent.AddListener
      (mitk::MessageDelegate1<Single3DViewWidget, const mitk::DataNode*>
      (this, &Single3DViewWidget::InternalNodeAdded ) );
  }

  mitk::BaseRenderer::GetInstance(m_RenderWindow->GetRenderWindow())->SetDataStorage(dataStorage);
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::InternalNodeAdded(const mitk::DataNode* node)
{
  this->NodeAdded(node);
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::InternalNodeRemoved (const mitk::DataNode* node)
{
  if (m_ImageNode.IsNotNull() && node == m_ImageNode)
  {
    this->SetImageNode(NULL);
  }
  this->NodeRemoved(node);
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::InternalNodeChanged(const mitk::DataNode* node)
{
  this->NodeChanged(node);
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* Single3DViewWidget::GetRenderWindow() const
{
  return m_RenderWindow;
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::EnableGradientBackground()
{
  m_GradientBackground->Enable();
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::DisableGradientBackground()
{
  m_GradientBackground->Disable();
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::SetGradientBackgroundColors( const mitk::Color & upper, const mitk::Color & lower )
{
  m_GradientBackground->SetGradientColors(upper[0], upper[1], upper[2], lower[0], lower[1], lower[2]);
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::EnableDepartmentLogo()
{
   m_LogoRendering->Enable();
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::DisableDepartmentLogo()
{
   m_LogoRendering->Disable();
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::SetDepartmentLogoPath(const QString& path)
{
  m_LogoRendering->SetLogoSource(qPrintable(path));
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::resizeEvent(QResizeEvent* /*event*/)
{
  this->Update();
}


//-----------------------------------------------------------------------------
void Single3DViewWidget::SetImageNode(mitk::DataNode* node)
{
  // Remember: node can be NULL, as we have to respond to NodeRemoved events.
  if (node == NULL)
  {
    m_Image = NULL;
    m_ImageNode = NULL;
  }
  else
  {
    mitk::Image* image = dynamic_cast<mitk::Image*>(node->GetData());
    if (image != NULL)
    {
      m_Image = image;
      m_ImageNode = const_cast<mitk::DataNode*>(node);
    }
  }
  this->Update();
}

} // end namespace

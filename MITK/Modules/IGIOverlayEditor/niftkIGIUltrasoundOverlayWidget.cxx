/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIUltrasoundOverlayWidget.h"
#include <mitkCoordinateAxesData.h>
#include <mitkNodePredicateDataType.h>
#include <mitkImage.h>
#include <mitkBaseRenderer.h>
#include <mitkRenderingManager.h>
#include <mitkTimeGeometry.h>
#include <mitkGlobalInteraction.h>
#include <mitkFocusManager.h>

namespace niftk
{

//-----------------------------------------------------------------------------
IGIUltrasoundOverlayWidget::IGIUltrasoundOverlayWidget(QWidget * /*parent*/)
{
  this->setupUi(this);

  m_3DViewer->GetRenderer()->SetMapperID(mitk::BaseRenderer::Standard3D );

  connect(m_3DViewCheckBox, SIGNAL(toggled(bool)), this, SLOT(On3DViewerCheckBoxChecked(bool)));
  connect(m_LeftImageCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnLeftOverlayCheckBoxChecked(bool)));
  connect(m_LeftImageCombo, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), this, SLOT(OnLeftImageSelected(const mitk::DataNode*)));

  m_LeftImageCombo->setCurrentIndex(0);
  m_LeftImageCheckBox->setChecked(true);
  m_3DViewCheckBox->setChecked(true);

  int width = m_Splitter->width();
  QList<int> sizes;
  sizes.append(width);
  sizes.append(width);
  m_Splitter->setSizes(sizes);
  m_Splitter->setChildrenCollapsible(true);
}


//-----------------------------------------------------------------------------
IGIUltrasoundOverlayWidget::~IGIUltrasoundOverlayWidget()
{
  this->DeRegisterDataStorageListeners();
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayWidget::DeRegisterDataStorageListeners()
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->ChangedNodeEvent.RemoveListener
      (mitk::MessageDelegate1<IGIUltrasoundOverlayWidget, const mitk::DataNode*>
      (this, &IGIUltrasoundOverlayWidget::NodeChanged ) );
  }
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayWidget::OnLeftOverlayCheckBoxChecked(bool checked)
{
  if (checked)
  {
    m_3DViewCheckBox->setEnabled(true);
    mitk::RenderingManager::GetInstance()->AddRenderWindow(m_LeftOverlayViewer->GetRenderWindow()->GetVtkRenderWindow());
  }
  else
  {
    m_3DViewCheckBox->setEnabled(false);
    mitk::RenderingManager::GetInstance()->RemoveRenderWindow(m_LeftOverlayViewer->GetRenderWindow()->GetVtkRenderWindow());
  }
  m_LeftOverlayViewer->setVisible(checked);
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayWidget::On3DViewerCheckBoxChecked(bool checked)
{
  if (checked)
  {
    m_LeftImageCheckBox->setEnabled(true);
    mitk::RenderingManager::GetInstance()->AddRenderWindow(m_3DViewer->GetRenderWindow());
  }
  else
  {
    m_LeftImageCheckBox->setEnabled(false);
    mitk::RenderingManager::GetInstance()->RemoveRenderWindow(m_3DViewer->GetRenderWindow());
  }
  m_3DViewer->setVisible(checked);
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayWidget::OnLeftImageSelected(const mitk::DataNode* node)
{
  m_LeftOverlayViewer->SetImageNode(node);
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayWidget::SetDataStorage(mitk::DataStorage* storage)
{
  if (m_DataStorage.IsNotNull() && m_DataStorage != storage)
  {
    this->DeRegisterDataStorageListeners();
  }

  m_DataStorage = storage;
  
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->ChangedNodeEvent.AddListener
      (mitk::MessageDelegate1<IGIUltrasoundOverlayWidget, const mitk::DataNode*>
      (this, &IGIUltrasoundOverlayWidget::NodeChanged ) );
  }
  
  mitk::TimeGeometry::Pointer geometry = storage->ComputeBoundingGeometry3D(storage->GetAll());
  mitk::RenderingManager::GetInstance()->InitializeView(m_3DViewer->GetVtkRenderWindow(), geometry);

  m_3DViewer->GetRenderer()->SetDataStorage(storage);
  m_LeftOverlayViewer->SetDataStorage(storage);
  m_LeftImageCombo->SetDataStorage(storage);

  mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
  m_LeftImageCombo->SetPredicate(isImage);
  m_LeftImageCombo->SetAutoSelectNewItems(false);

  m_LeftImageCombo->setCurrentIndex(0);
  this->OnLeftImageSelected(nullptr);
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayWidget::NodeChanged(const mitk::DataNode* node)
{
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* IGIUltrasoundOverlayWidget::GetActiveQmitkRenderWindow() const
{
  QmitkRenderWindow *result = NULL;

  mitk::FocusManager *focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    mitk::BaseRenderer *renderer = focusManager->GetFocused();
    if (m_LeftOverlayViewer->GetRenderWindow()->GetRenderer() == renderer)
    {
      result = m_LeftOverlayViewer->GetRenderWindow();
    }
    else if (m_3DViewer->GetRenderer() == renderer)
    {
      result = m_3DViewer;
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
QHash<QString, QmitkRenderWindow *> IGIUltrasoundOverlayWidget::GetQmitkRenderWindows() const
{
  QHash<QString, QmitkRenderWindow *> result;
  result.insert("overlay", m_LeftOverlayViewer->GetRenderWindow());
  result.insert("3d", m_3DViewer);
  return result;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* IGIUltrasoundOverlayWidget::GetQmitkRenderWindow(const QString &id) const
{
  QmitkRenderWindow *result = NULL;
  if (id == "3d")
  {
    result = m_3DViewer;
  }
  else if (id == "overlay")
  {
    result =  m_LeftOverlayViewer->GetRenderWindow();
  }
  return result;
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayWidget::SetDepartmentLogoPath(const QString& path)
{
  m_LeftOverlayViewer->SetDepartmentLogoPath(path);
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayWidget::EnableDepartmentLogo()
{
  m_LeftOverlayViewer->EnableDepartmentLogo();
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayWidget::DisableDepartmentLogo()
{
  m_LeftOverlayViewer->DisableDepartmentLogo();
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayWidget::SetGradientBackgroundColors(const mitk::Color& colour1, const mitk::Color& colour2)
{
  m_LeftOverlayViewer->SetGradientBackgroundColors(colour1, colour2);
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayWidget::EnableGradientBackground()
{
  m_LeftOverlayViewer->EnableGradientBackground();
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayWidget::DisableGradientBackground()
{
  m_LeftOverlayViewer->DisableGradientBackground();
}


//-----------------------------------------------------------------------------
void IGIUltrasoundOverlayWidget::Update()
{
  m_LeftOverlayViewer->Update();
}

} // end namespace

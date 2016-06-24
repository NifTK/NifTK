/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIOverlayEditorWidget.h"
#include <mitkCoordinateAxesData.h>
#include <mitkTrackedImage.h>

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
IGIOverlayEditorWidget::IGIOverlayEditorWidget(QWidget * /*parent*/)
{
  this->setupUi(this);

  m_OpacitySlider->setMinimum(0);
  m_OpacitySlider->setMaximum(100);
  m_OpacitySlider->setSingleStep(1);
  m_OpacitySlider->setPageStep(10);
  m_OpacitySlider->setValue(static_cast<int>(m_OverlayViewer->GetOpacity()*100));

  m_3DViewer->GetRenderer()->SetMapperID(mitk::BaseRenderer::Standard3D );

  m_OverlayViewer->setVisible(true);
  m_3DViewer->setVisible(true);

  m_OverlayCheckBox->setChecked(true);
  m_3DViewerCheckBox->setChecked(true);

  connect(m_OverlayCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnOverlayCheckBoxChecked(bool)));
  connect(m_3DViewerCheckBox, SIGNAL(toggled(bool)), this, SLOT(On3DViewerCheckBoxChecked(bool)));
  connect(m_ImageCombo, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), this, SLOT(OnImageSelected(const mitk::DataNode*)));
  connect(m_TransformCombo, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), this, SLOT(OnTransformSelected(const mitk::DataNode*)));
  connect(m_OpacitySlider, SIGNAL(sliderMoved(int)), this, SLOT(OnOpacitySliderMoved(int)));

  int width = m_Splitter->width();
  QList<int> sizes;
  sizes.append(width);
  sizes.append(width);
  m_Splitter->setSizes(sizes);
}


//-----------------------------------------------------------------------------
IGIOverlayEditorWidget::~IGIOverlayEditorWidget()
{
  this->DeRegisterDataStorageListeners();
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorWidget::DeRegisterDataStorageListeners()
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->ChangedNodeEvent.RemoveListener
      (mitk::MessageDelegate1<IGIOverlayEditorWidget, const mitk::DataNode*>
      (this, &IGIOverlayEditorWidget::NodeChanged ) );
  }
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorWidget::OnOverlayCheckBoxChecked(bool checked)
{
  if (!checked)
  {
    m_3DViewerCheckBox->setEnabled(false);
    mitk::RenderingManager::GetInstance()->RemoveRenderWindow(m_OverlayViewer->GetRenderWindow()->GetVtkRenderWindow());
  }
  else
  {
    m_3DViewerCheckBox->setEnabled(true);
    mitk::RenderingManager::GetInstance()->AddRenderWindow(m_OverlayViewer->GetRenderWindow()->GetVtkRenderWindow());
  }
  m_OverlayViewer->setVisible(checked);
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorWidget::On3DViewerCheckBoxChecked(bool checked)
{
  if (!checked)
  {
    m_OverlayCheckBox->setEnabled(false);
    mitk::RenderingManager::GetInstance()->RemoveRenderWindow(m_3DViewer->GetRenderWindow());
  }
  else
  {
    m_OverlayCheckBox->setEnabled(true);
    mitk::RenderingManager::GetInstance()->AddRenderWindow(m_3DViewer->GetRenderWindow());
  }
  m_3DViewer->setVisible(checked);
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorWidget::OnOpacitySliderMoved(int value)
{
  m_OverlayViewer->SetOpacity(value / 100.0);
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorWidget::OnImageSelected(const mitk::DataNode* node)
{
  m_OverlayViewer->SetImageNode(node);
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorWidget::OnTransformSelected(const mitk::DataNode* node)
{
  m_OverlayViewer->SetTransformNode(node);
  mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorWidget::SetDataStorage(mitk::DataStorage* storage)
{
  if (m_DataStorage.IsNotNull() && m_DataStorage != storage)
  {
    this->DeRegisterDataStorageListeners();
  }

  m_DataStorage = storage;
  
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->ChangedNodeEvent.AddListener
      (mitk::MessageDelegate1<IGIOverlayEditorWidget, const mitk::DataNode*>
      (this, &IGIOverlayEditorWidget::NodeChanged ) );
  }
  
  mitk::TimeGeometry::Pointer geometry = storage->ComputeBoundingGeometry3D(storage->GetAll());
  mitk::RenderingManager::GetInstance()->InitializeView(m_3DViewer->GetVtkRenderWindow(), geometry);

  m_3DViewer->GetRenderer()->SetDataStorage(storage);
  m_OverlayViewer->SetDataStorage(storage);
  m_ImageCombo->SetDataStorage(storage);
  m_TransformCombo->SetDataStorage(storage);

  mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
  m_ImageCombo->SetPredicate(isImage);
  m_ImageCombo->SetAutoSelectNewItems(false);

  mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::Pointer isTransform = mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::New();
  m_TransformCombo->SetPredicate(isTransform);
  m_TransformCombo->SetAutoSelectNewItems(false);
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorWidget::NodeChanged(const mitk::DataNode* node)
{
  bool propValue = false;
  if (node != NULL
    && !m_OverlayViewer->GetCameraTrackingMode()
    && node->GetBoolProperty(mitk::TrackedImage::TRACKED_IMAGE_SELECTED_PROPERTY_NAME, propValue)
    && propValue)
  {
    m_ImageCombo->SetSelectedNode(const_cast<mitk::DataNode*>(node));
  }
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* IGIOverlayEditorWidget::GetActiveQmitkRenderWindow() const
{
  QmitkRenderWindow *result = NULL;

  mitk::FocusManager *focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  if (focusManager != NULL)
  {
    mitk::BaseRenderer *renderer = focusManager->GetFocused();
    if (m_OverlayViewer->GetRenderWindow()->GetRenderer() == renderer)
    {
      result = m_OverlayViewer->GetRenderWindow();
    }
    else if (m_3DViewer->GetRenderer() == renderer)
    {
      result = m_3DViewer;
    }
  }
  return result;
}


//-----------------------------------------------------------------------------
QHash<QString, QmitkRenderWindow *> IGIOverlayEditorWidget::GetQmitkRenderWindows() const
{
  QHash<QString, QmitkRenderWindow *> result;
  result.insert("overlay", m_OverlayViewer->GetRenderWindow());
  result.insert("3d", m_3DViewer);
  return result;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* IGIOverlayEditorWidget::GetQmitkRenderWindow(const QString &id) const
{
  QmitkRenderWindow *result = NULL;
  if (id == "3d")
  {
    result = m_3DViewer;
  }
  else if (id == "overlay")
  {
    result =  m_OverlayViewer->GetRenderWindow();
  }
  return result;
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorWidget::SetCalibrationFileName(const QString& fileName)
{
  m_OverlayViewer->SetTrackingCalibrationFileName(fileName);
}


//-----------------------------------------------------------------------------
QString IGIOverlayEditorWidget::GetCalibrationFileName() const
{
  return m_OverlayViewer->GetTrackingCalibrationFileName();
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorWidget::SetCameraTrackingMode(const bool& isCameraTracking)
{
  m_OverlayViewer->SetCameraTrackingMode(isCameraTracking);
  m_TransformCombo->setVisible(isCameraTracking);
  m_TransformLabel->setVisible(isCameraTracking);
  m_ImageCombo->setVisible(isCameraTracking);
  m_ImageLabel->setVisible(isCameraTracking);
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorWidget::SetClipToImagePlane(const bool& clipToImagePlane)
{
  m_OverlayViewer->SetClipToImagePlane(clipToImagePlane);
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorWidget::SetDepartmentLogoPath(const QString& path)
{
  m_OverlayViewer->SetDepartmentLogoPath(path);
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorWidget::EnableDepartmentLogo()
{
  m_OverlayViewer->EnableDepartmentLogo();
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorWidget::DisableDepartmentLogo()
{
  m_OverlayViewer->DisableDepartmentLogo();
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorWidget::SetGradientBackgroundColors(const mitk::Color& colour1, const mitk::Color& colour2)
{
  m_OverlayViewer->SetGradientBackgroundColors(colour1, colour2);
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorWidget::EnableGradientBackground()
{
  m_OverlayViewer->EnableGradientBackground();
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorWidget::DisableGradientBackground()
{
  m_OverlayViewer->DisableGradientBackground();
}


//-----------------------------------------------------------------------------
void IGIOverlayEditorWidget::Update()
{
  m_OverlayViewer->Update();
}

} // end namespace

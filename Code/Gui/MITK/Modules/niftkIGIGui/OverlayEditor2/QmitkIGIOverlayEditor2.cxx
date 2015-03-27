/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGIOverlayEditor2.h"
#include <mitkNodePredicateDataType.h>
#include <mitkImage.h>
#include <mitkBaseRenderer.h>
#include <mitkRenderingManager.h>
#include <mitkTimeGeometry.h>
#include <mitkCoordinateAxesData.h>
#include <mitkGlobalInteraction.h>
#include <mitkFocusManager.h>
#include <mitkTrackedImage.h>

//-----------------------------------------------------------------------------
QmitkIGIOverlayEditor2::QmitkIGIOverlayEditor2(QWidget * /*parent*/)
{
  this->setupUi(this);

  m_OpacitySlider->setMinimum(0);
  m_OpacitySlider->setMaximum(100);
  m_OpacitySlider->setSingleStep(1);
  m_OpacitySlider->setPageStep(10);
//  m_OpacitySlider->setValue(static_cast<int>(m_OverlayViewer->GetOpacity()*100));

//  m_3DViewer->GetRenderer()->SetMapperID(mitk::BaseRenderer::Standard3D );

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
QmitkIGIOverlayEditor2::~QmitkIGIOverlayEditor2()
{
  this->DeRegisterDataStorageListeners();
}


//-----------------------------------------------------------------------------
void QmitkIGIOverlayEditor2::SetOclResourceService(OclResourceService* oclserv)
{
  m_OverlayViewer->SetOclResourceService(oclserv);
  m_3DViewer->SetOclResourceService(oclserv);
}


//-----------------------------------------------------------------------------
void QmitkIGIOverlayEditor2::DeRegisterDataStorageListeners()
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->ChangedNodeEvent.RemoveListener
      (mitk::MessageDelegate1<QmitkIGIOverlayEditor2, const mitk::DataNode*>
      (this, &QmitkIGIOverlayEditor2::NodeChanged ) );
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIOverlayEditor2::OnOverlayCheckBoxChecked(bool checked)
{
  if (!checked)
  {
    m_3DViewerCheckBox->setEnabled(false);
    //mitk::RenderingManager::GetInstance()->RemoveRenderWindow(m_OverlayViewer->GetRenderWindow()->GetVtkRenderWindow());
  }
  else
  {
    m_3DViewerCheckBox->setEnabled(true);
    //mitk::RenderingManager::GetInstance()->AddRenderWindow(m_OverlayViewer->GetRenderWindow()->GetVtkRenderWindow());
  }
  m_OverlayViewer->setVisible(checked);
}


//-----------------------------------------------------------------------------
void QmitkIGIOverlayEditor2::On3DViewerCheckBoxChecked(bool checked)
{
  if (!checked)
  {
    m_OverlayCheckBox->setEnabled(false);
    //mitk::RenderingManager::GetInstance()->RemoveRenderWindow(m_3DViewer->GetRenderWindow());
  }
  else
  {
    m_OverlayCheckBox->setEnabled(true);
    //mitk::RenderingManager::GetInstance()->AddRenderWindow(m_3DViewer->GetRenderWindow());
  }
  m_3DViewer->setVisible(checked);
}


//-----------------------------------------------------------------------------
void QmitkIGIOverlayEditor2::OnOpacitySliderMoved(int value)
{
  //m_OverlayViewer->SetOpacity(value / 100.0);
  //mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void QmitkIGIOverlayEditor2::OnImageSelected(const mitk::DataNode* node)
{
  //m_OverlayViewer->SetImageNode(node);
  //mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void QmitkIGIOverlayEditor2::OnTransformSelected(const mitk::DataNode* node)
{
  //m_OverlayViewer->SetTransformNode(node);
  //mitk::RenderingManager::GetInstance()->RequestUpdateAll();
}


//-----------------------------------------------------------------------------
void QmitkIGIOverlayEditor2::SetDataStorage(mitk::DataStorage* storage)
{
  if (m_DataStorage.IsNotNull() && m_DataStorage != storage)
  {
    this->DeRegisterDataStorageListeners();
  }

  m_DataStorage = storage;
  
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->ChangedNodeEvent.AddListener
      (mitk::MessageDelegate1<QmitkIGIOverlayEditor2, const mitk::DataNode*>
      (this, &QmitkIGIOverlayEditor2::NodeChanged ) );
  }
  
  //mitk::TimeGeometry::Pointer geometry = storage->ComputeBoundingGeometry3D(storage->GetAll());
  //mitk::RenderingManager::GetInstance()->InitializeView(m_3DViewer->GetVtkRenderWindow(), geometry);

  m_3DViewer->SetDataStorage(storage);
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
void QmitkIGIOverlayEditor2::NodeChanged(const mitk::DataNode* node)
{
  //bool propValue = false;
  //if (node != NULL
  //  && !m_OverlayViewer->GetCameraTrackingMode()
  //  && node->GetBoolProperty(mitk::TrackedImage::TRACKED_IMAGE_SELECTED_PROPERTY_NAME, propValue)
  //  && propValue)
  //{
  //  m_ImageCombo->SetSelectedNode(const_cast<mitk::DataNode*>(node));
  //}
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* QmitkIGIOverlayEditor2::GetActiveQmitkRenderWindow() const
{
  QmitkRenderWindow *result = NULL;

  //mitk::FocusManager *focusManager = mitk::GlobalInteraction::GetInstance()->GetFocusManager();
  //if (focusManager != NULL)
  //{
  //  mitk::BaseRenderer *renderer = focusManager->GetFocused();
  //  if (m_OverlayViewer->GetRenderWindow()->GetRenderer() == renderer)
  //  {
  //    result = m_OverlayViewer->GetRenderWindow();
  //  }
  //  else if (m_3DViewer->GetRenderer() == renderer)
  //  {
  //    result = m_3DViewer;
  //  }
  //}
  return result;
}


//-----------------------------------------------------------------------------
QHash<QString, QmitkRenderWindow *> QmitkIGIOverlayEditor2::GetQmitkRenderWindows() const
{
  QHash<QString, QmitkRenderWindow *> result;
  //result.insert("overlay", m_OverlayViewer->GetRenderWindow());
  //result.insert("3d", m_3DViewer);
  return result;
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* QmitkIGIOverlayEditor2::GetQmitkRenderWindow(const QString &id) const
{
  QmitkRenderWindow *result = NULL;
  //if (id == "3d")
  //{
  //  result = m_3DViewer;
  //}
  //else if (id == "overlay")
  //{
  //  result =  m_OverlayViewer->GetRenderWindow();
  //}
  return result;
}


//-----------------------------------------------------------------------------
void QmitkIGIOverlayEditor2::SetCalibrationFileName(const std::string& fileName)
{
  //m_OverlayViewer->SetTrackingCalibrationFileName(fileName);
}


//-----------------------------------------------------------------------------
std::string QmitkIGIOverlayEditor2::GetCalibrationFileName() const
{
  return "";//m_OverlayViewer->GetTrackingCalibrationFileName();
}


//-----------------------------------------------------------------------------
//void QmitkIGIOverlayEditor2::SetCameraTrackingMode(const bool& isCameraTracking)
//{
//  //m_OverlayViewer->SetCameraTrackingMode(isCameraTracking);
//  m_TransformCombo->setVisible(isCameraTracking);
//  m_TransformLabel->setVisible(isCameraTracking);
//  m_ImageCombo->setVisible(isCameraTracking);
//  m_ImageLabel->setVisible(isCameraTracking);
//}


//-----------------------------------------------------------------------------
//void QmitkIGIOverlayEditor2::SetClipToImagePlane(const bool& clipToImagePlane)
//{
//  m_OverlayViewer->SetClipToImagePlane(clipToImagePlane);
//}


//-----------------------------------------------------------------------------
//void QmitkIGIOverlayEditor2::SetDepartmentLogoPath(const std::string path)
//{
//  m_OverlayViewer->SetDepartmentLogoPath(path.c_str());
//}


//-----------------------------------------------------------------------------
//void QmitkIGIOverlayEditor2::EnableDepartmentLogo()
//{
//  m_OverlayViewer->EnableDepartmentLogo();
//}


//-----------------------------------------------------------------------------
//void QmitkIGIOverlayEditor2::DisableDepartmentLogo()
//{
//  m_OverlayViewer->DisableDepartmentLogo();
//}


//-----------------------------------------------------------------------------
//void QmitkIGIOverlayEditor2::SetGradientBackgroundColors(const mitk::Color& colour1, const mitk::Color& colour2)
//{
//  m_OverlayViewer->SetGradientBackgroundColors(colour1, colour2);
//}


//-----------------------------------------------------------------------------
//void QmitkIGIOverlayEditor2::EnableGradientBackground()
//{
//  m_OverlayViewer->EnableGradientBackground();
//}


//-----------------------------------------------------------------------------
//void QmitkIGIOverlayEditor2::DisableGradientBackground()
//{
//  m_OverlayViewer->DisableGradientBackground();
//}


//-----------------------------------------------------------------------------
void QmitkIGIOverlayEditor2::Update()
{
  //m_OverlayViewer->Update();
}

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkIGIVLEditor.h"
#include <mitkNodePredicateDataType.h>
#include <mitkImage.h>
#include <mitkBaseRenderer.h>
#include <mitkRenderingManager.h>
#include <mitkTimeGeometry.h>
#include <mitkCoordinateAxesData.h>
#include <mitkGlobalInteraction.h>
#include <mitkFocusManager.h>

#include "VLQt4Widget.h"
#include <Rendering/SharedOGLContext.h>


//-----------------------------------------------------------------------------
QmitkIGIVLEditor::QmitkIGIVLEditor(QWidget * /*parent*/)
  : m_OverlayViewer(0)
  , m_3DViewer(0)
{
  this->setupUi(this);

  m_OverlayViewer = new VLQt4Widget(m_Splitter, SharedOGLContext::GetShareWidget());
  m_Splitter->addWidget(m_OverlayViewer);
  m_3DViewer = new VLQt4Widget(m_Splitter, SharedOGLContext::GetShareWidget());
  m_Splitter->addWidget(m_3DViewer);

  m_OpacitySlider->setMinimum(0);
  m_OpacitySlider->setMaximum(100);
  m_OpacitySlider->setSingleStep(1);
  m_OpacitySlider->setPageStep(10);
//  m_OpacitySlider->setValue(static_cast<int>(m_OverlayViewer->GetOpacity()*100));

  m_OverlayViewer->setVisible(true);
  m_OverlayViewer->setObjectName("QmitkIGIVLEditor::m_OverlayViewer");
  m_3DViewer->setVisible(true);
  m_3DViewer->setObjectName("QmitkIGIVLEditor::m_3DViewer");

  m_OverlayCheckBox->setChecked(true);
  m_3DViewerCheckBox->setChecked(true);

  bool ok = false;
  ok = QObject::connect(m_OverlayCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnOverlayCheckBoxChecked(bool)), Qt::QueuedConnection);
  assert(ok);
  ok = QObject::connect(m_3DViewerCheckBox, SIGNAL(toggled(bool)), this, SLOT(On3DViewerCheckBoxChecked(bool)), Qt::QueuedConnection);
  assert(ok);
  ok = QObject::connect(m_ImageCombo, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), this, SLOT(OnImageSelected(const mitk::DataNode*)), Qt::QueuedConnection);
  assert(ok);
  ok = QObject::connect(m_TransformCombo, SIGNAL(OnSelectionChanged(const mitk::DataNode*)), this, SLOT(OnTransformSelected(const mitk::DataNode*)), Qt::QueuedConnection);
  assert(ok);
  ok = QObject::connect(m_OpacitySlider, SIGNAL(sliderMoved(int)), this, SLOT(OnOpacitySliderMoved(int)), Qt::QueuedConnection);
  assert(ok);

  int width = m_Splitter->width();
  QList<int> sizes;
  sizes.append(width);
  sizes.append(width);
  m_Splitter->setSizes(sizes);
}


//-----------------------------------------------------------------------------
QmitkIGIVLEditor::~QmitkIGIVLEditor()
{
  this->DeRegisterDataStorageListeners();
}


//-----------------------------------------------------------------------------
void QmitkIGIVLEditor::SetBackgroundColour(unsigned int aabbggrr)
{
  float   r = (aabbggrr & 0xFF) / 255.0f;
  float   g = ((aabbggrr & 0xFF00) >> 8) / 255.0f;
  float   b = ((aabbggrr & 0xFF0000) >> 16) / 255.0f;
  m_OverlayViewer->SetBackgroundColour(r, g, b);
  m_3DViewer->SetBackgroundColour(r, g, b);
}


//-----------------------------------------------------------------------------
void QmitkIGIVLEditor::SetOclResourceService(OclResourceService* oclserv)
{
  m_OverlayViewer->SetOclResourceService(oclserv);
  m_3DViewer->SetOclResourceService(oclserv);
}


//-----------------------------------------------------------------------------
void QmitkIGIVLEditor::DeRegisterDataStorageListeners()
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->ChangedNodeEvent.RemoveListener
      (mitk::MessageDelegate1<QmitkIGIVLEditor, const mitk::DataNode*>
      (this, &QmitkIGIVLEditor::NodeChanged ) );
  }
}


//-----------------------------------------------------------------------------
void QmitkIGIVLEditor::OnOverlayCheckBoxChecked(bool checked)
{
  if (!checked)
  {
    m_3DViewerCheckBox->setEnabled(false);
  }
  else
  {
    m_3DViewerCheckBox->setEnabled(true);
  }
  m_OverlayViewer->setVisible(checked);
}


//-----------------------------------------------------------------------------
void QmitkIGIVLEditor::On3DViewerCheckBoxChecked(bool checked)
{
  if (!checked)
  {
    m_OverlayCheckBox->setEnabled(false);
  }
  else
  {
    m_OverlayCheckBox->setEnabled(true);
  }
  m_3DViewer->setVisible(checked);
}


//-----------------------------------------------------------------------------
void QmitkIGIVLEditor::OnOpacitySliderMoved(int value)
{
  //m_OverlayViewer->SetOpacity(value / 100.0);
}


//-----------------------------------------------------------------------------
void QmitkIGIVLEditor::OnImageSelected(const mitk::DataNode* node)
{
  m_OverlayViewer->SetBackgroundNode(node);
}


//-----------------------------------------------------------------------------
void QmitkIGIVLEditor::OnTransformSelected(const mitk::DataNode* node)
{
  m_OverlayViewer->SetCameraTrackingNode(node);
}


//-----------------------------------------------------------------------------
void QmitkIGIVLEditor::SetDataStorage(mitk::DataStorage* storage)
{
  if (m_DataStorage.IsNotNull() && m_DataStorage != storage)
  {
    this->DeRegisterDataStorageListeners();
  }

  m_DataStorage = storage;
  
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->ChangedNodeEvent.AddListener
      (mitk::MessageDelegate1<QmitkIGIVLEditor, const mitk::DataNode*>
      (this, &QmitkIGIVLEditor::NodeChanged ) );
  }

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
void QmitkIGIVLEditor::NodeChanged(const mitk::DataNode* node)
{
}


//-----------------------------------------------------------------------------
void QmitkIGIVLEditor::Update()
{
  //m_OverlayViewer->Update();
}

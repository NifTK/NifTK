/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkVLVideoOverlayWidget.h"
#include <niftkVLWidget.h>
#include <niftkSharedOGLContext.h>
#include <mitkNodePredicateDataType.h>
#include <mitkImage.h>

#include <niftkCoordinateAxesData.h>

namespace niftk
{

//-----------------------------------------------------------------------------
VLVideoOverlayWidget::VLVideoOverlayWidget(QWidget * /*parent*/)
: m_HorizontalLayout(nullptr)
, m_OverlayViewers(nullptr)
, m_LeftOverlayViewer(nullptr)
, m_RightOverlayViewer(nullptr)
, m_TrackedViewer(nullptr)
, m_3DViewer(nullptr)
{
  this->setupUi(this);

  m_OverlayViewers = new QWidget(m_Splitter);
  m_HorizontalLayout = new QHBoxLayout(m_OverlayViewers);
  m_HorizontalLayout->setContentsMargins(0, 0, 0, 0);
  m_LeftOverlayViewer = new VLWidget(m_OverlayViewers, niftk::SharedOGLContext::GetShareWidget());
  m_LeftOverlayViewer->setObjectName("VLVideoOverlayWidget::m_LeftOverlayViewer");
  m_HorizontalLayout->addWidget(m_LeftOverlayViewer);
  m_RightOverlayViewer = new VLWidget(m_OverlayViewers, niftk::SharedOGLContext::GetShareWidget());
  m_RightOverlayViewer->setObjectName("VLVideoOverlayWidget::m_RightOverlayViewer");
  m_HorizontalLayout->addWidget(m_RightOverlayViewer);
  m_TrackedViewer = new VLWidget(m_OverlayViewers, niftk::SharedOGLContext::GetShareWidget());
  m_TrackedViewer->setObjectName("VLVideoOverlayWidget::m_TrackedViewer");
  m_HorizontalLayout->addWidget(m_TrackedViewer);

  m_Splitter->addWidget(m_OverlayViewers);
  m_3DViewer = new VLWidget(m_Splitter, niftk::SharedOGLContext::GetShareWidget());
  m_3DViewer->setObjectName("VLVideoOverlayWidget::m_3DViewer");
  m_Splitter->addWidget(m_3DViewer);

  m_OpacitySlider->setMinimum(0);
  m_OpacitySlider->setMaximum(100);
  m_OpacitySlider->setSingleStep(1);
  m_OpacitySlider->setPageStep(10);
  m_OpacitySlider->setValue(100);

  bool ok = false;
  ok = connect(m_3DViewCheckBox, SIGNAL(toggled(bool)), this, SLOT(On3DViewerCheckBoxChecked(bool)));
  assert(ok);
  ok = connect(m_LeftImageCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnLeftOverlayCheckBoxChecked(bool)));
  assert(ok);
  ok = connect(m_RightImageCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnRightOverlayCheckBoxChecked(bool)));
  assert(ok);
  ok = connect(m_TrackedViewCheckBox, SIGNAL(toggled(bool)), this, SLOT(OnTrackedViewerCheckBoxChecked(bool)));
  assert(ok);
  ok = connect(m_OpacitySlider, SIGNAL(sliderMoved(int)), this, SLOT(OnOpacitySliderMoved(int)));
  assert(ok);

  m_LeftImageCheckBox->setChecked(true);
  m_LeftOverlayViewer->setVisible(true);
  m_RightImageCheckBox->setChecked(false);
  m_RightOverlayViewer->setVisible(false);
  m_TrackedViewCheckBox->setChecked(false);
  m_TrackedViewer->setVisible(false);
  m_3DViewCheckBox->setChecked(true);
  m_3DViewer->setVisible(true);

  int width = m_Splitter->width();
  QList<int> sizes;
  sizes.append(width);
  sizes.append(width);
  m_Splitter->setSizes(sizes);
}


//-----------------------------------------------------------------------------
VLVideoOverlayWidget::~VLVideoOverlayWidget()
{
  this->DeRegisterDataStorageListeners();
}


//-----------------------------------------------------------------------------
void VLVideoOverlayWidget::SetBackgroundColour(unsigned int aabbggrr)
{
  float   r = (aabbggrr & 0xFF) / 255.0f;
  float   g = ((aabbggrr & 0xFF00) >> 8) / 255.0f;
  float   b = ((aabbggrr & 0xFF0000) >> 16) / 255.0f;
  m_LeftOverlayViewer->vlSceneView()->setBackgroundColor(r, g, b);
  m_RightOverlayViewer->vlSceneView()->setBackgroundColor(r, g, b);
  m_TrackedViewer->vlSceneView()->setBackgroundColor(r, g, b);
  m_3DViewer->vlSceneView()->setBackgroundColor(r, g, b);
}


//-----------------------------------------------------------------------------
void VLVideoOverlayWidget::DeRegisterDataStorageListeners()
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->ChangedNodeEvent.RemoveListener
      (mitk::MessageDelegate1<VLVideoOverlayWidget, const mitk::DataNode*>
      (this, &VLVideoOverlayWidget::NodeChanged ) );
  }
}


//-----------------------------------------------------------------------------
void VLVideoOverlayWidget::OnLeftOverlayCheckBoxChecked(bool checked)
{
  m_LeftOverlayViewer->setVisible(checked);
  m_OverlayViewers->setVisible(   m_LeftImageCheckBox->isChecked()
                               || m_RightImageCheckBox->isChecked()
                               || m_TrackedViewCheckBox->isChecked());
}


//-----------------------------------------------------------------------------
void VLVideoOverlayWidget::OnRightOverlayCheckBoxChecked(bool checked)
{
  m_RightOverlayViewer->setVisible(checked);
  m_OverlayViewers->setVisible(   m_LeftImageCheckBox->isChecked()
                               || m_RightImageCheckBox->isChecked()
                               || m_TrackedViewCheckBox->isChecked());
}


//-----------------------------------------------------------------------------
void VLVideoOverlayWidget::On3DViewerCheckBoxChecked(bool checked)
{
  m_3DViewer->setVisible(checked);
}


//-----------------------------------------------------------------------------
void VLVideoOverlayWidget::OnTrackedViewerCheckBoxChecked(bool checked)
{
  m_TrackedViewer->setVisible(checked);
  m_OverlayViewers->setVisible(   m_LeftImageCheckBox->isChecked()
                               || m_RightImageCheckBox->isChecked()
                               || m_TrackedViewCheckBox->isChecked());
}


//-----------------------------------------------------------------------------
void VLVideoOverlayWidget::OnOpacitySliderMoved(int value)
{
  m_LeftOverlayViewer->vlSceneView()->setOpacity(static_cast<float>(value)/100.0);
  m_RightOverlayViewer->vlSceneView()->setOpacity(static_cast<float>(value)/100.0);
}


//-----------------------------------------------------------------------------
void VLVideoOverlayWidget::OnLeftImageSelected(const mitk::DataNode* node)
{
  if (node != nullptr)
  {
    m_LeftOverlayViewer->vlSceneView()->setBackgroundNode(node);
    m_TrackedViewer->vlSceneView()->setBackgroundNode(node);
  }
}


//-----------------------------------------------------------------------------
void VLVideoOverlayWidget::OnRightImageSelected(const mitk::DataNode* node)
{
  if (node != nullptr)
  {
    m_RightOverlayViewer->vlSceneView()->setBackgroundNode(node);
  }
}


//-----------------------------------------------------------------------------
void VLVideoOverlayWidget::OnTransformSelected(const mitk::DataNode* node)
{
  if (node != nullptr)
  {
    m_LeftOverlayViewer->vlSceneView()->setCameraTrackingNode(node);
    m_RightOverlayViewer->vlSceneView()->setCameraTrackingNode(node);
    m_TrackedViewer->vlSceneView()->setCameraTrackingNode(node);
  }
}


//-----------------------------------------------------------------------------
void VLVideoOverlayWidget::SetEyeHandFileName(const std::string& fileName)
{
  m_3DViewer->vlSceneView()->setEyeHandFileName(fileName);
  m_LeftOverlayViewer->vlSceneView()->setEyeHandFileName(fileName);
  m_RightOverlayViewer->vlSceneView()->setEyeHandFileName(fileName);
  m_TrackedViewer->vlSceneView()->setEyeHandFileName(fileName);
}


//-----------------------------------------------------------------------------
void VLVideoOverlayWidget::SetDataStorage(mitk::DataStorage* storage)
{
  if (m_DataStorage.IsNotNull() && m_DataStorage != storage)
  {
    this->DeRegisterDataStorageListeners();
  }

  m_DataStorage = storage;

  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->ChangedNodeEvent.AddListener
      (mitk::MessageDelegate1<VLVideoOverlayWidget, const mitk::DataNode*>
      (this, &VLVideoOverlayWidget::NodeChanged ) );
  }

  m_3DViewer->vlSceneView()->setDataStorage(storage);
  m_LeftOverlayViewer->vlSceneView()->setDataStorage(storage);
  m_RightOverlayViewer->vlSceneView()->setDataStorage(storage);
  m_TrackedViewer->vlSceneView()->setDataStorage(storage);

  mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage =
      mitk::TNodePredicateDataType<mitk::Image>::New();

  m_LeftImageCombo->SetAutoSelectNewItems(false);
  m_LeftImageCombo->SetPredicate(isImage);
  m_LeftImageCombo->SetDataStorage(storage);
  m_LeftImageCombo->setCurrentIndex(0);

  m_RightImageCombo->SetAutoSelectNewItems(false);
  m_RightImageCombo->SetPredicate(isImage);
  m_RightImageCombo->SetDataStorage(storage);
  m_RightImageCombo->setCurrentIndex(0);

  mitk::TNodePredicateDataType<CoordinateAxesData>::Pointer isTransform =
      mitk::TNodePredicateDataType<CoordinateAxesData>::New();

  m_TrackingCombo->SetAutoSelectNewItems(false);
  m_TrackingCombo->SetPredicate(isTransform);
  m_TrackingCombo->SetDataStorage(storage);
  m_TrackingCombo->setCurrentIndex(0);

  bool ok = false;
  ok = connect(m_LeftImageCombo, SIGNAL(OnSelectionChanged(const mitk::DataNode*)),
               this, SLOT(OnLeftImageSelected(const mitk::DataNode*)));
  assert(ok);
  ok = connect(m_RightImageCombo, SIGNAL(OnSelectionChanged(const mitk::DataNode*)),
               this, SLOT(OnRightImageSelected(const mitk::DataNode*)));
  assert(ok);
  ok = connect(m_TrackingCombo, SIGNAL(OnSelectionChanged(const mitk::DataNode*)),
               this, SLOT(OnTransformSelected(const mitk::DataNode*)));
  assert(ok);
}


//-----------------------------------------------------------------------------
void VLVideoOverlayWidget::NodeChanged(const mitk::DataNode* node)
{
}

} // end namespace

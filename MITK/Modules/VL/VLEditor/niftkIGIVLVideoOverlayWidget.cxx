/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkIGIVLVideoOverlayWidget.h"
#include <VLQtWidget.h>
#include <niftkSharedOGLContext.h>
#include <mitkNodePredicateDataType.h>
#include <mitkImage.h>
#include <mitkCoordinateAxesData.h>

namespace niftk
{

//-----------------------------------------------------------------------------
IGIVLVideoOverlayWidget::IGIVLVideoOverlayWidget(QWidget * /*parent*/)
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
  m_LeftOverlayViewer = new VLQtWidget(m_OverlayViewers, niftk::SharedOGLContext::GetShareWidget());
  m_LeftOverlayViewer->setObjectName("IGIVLVideoOverlayWidget::m_LeftOverlayViewer");
  m_HorizontalLayout->addWidget(m_LeftOverlayViewer);
  m_RightOverlayViewer = new VLQtWidget(m_OverlayViewers, niftk::SharedOGLContext::GetShareWidget());
  m_RightOverlayViewer->setObjectName("IGIVLVideoOverlayWidget::m_RightOverlayViewer");
  m_HorizontalLayout->addWidget(m_RightOverlayViewer);
  m_TrackedViewer = new VLQtWidget(m_OverlayViewers, niftk::SharedOGLContext::GetShareWidget());
  m_TrackedViewer->setObjectName("IGIVLVideoOverlayWidget::m_TrackedViewer");
  m_HorizontalLayout->addWidget(m_TrackedViewer);

  m_Splitter->addWidget(m_OverlayViewers);
  m_3DViewer = new VLQtWidget(m_Splitter, niftk::SharedOGLContext::GetShareWidget());
  m_3DViewer->setObjectName("IGIVLVideoOverlayWidget::m_3DViewer");
  m_Splitter->addWidget(m_3DViewer);

  m_OpacitySlider->setMinimum(0);
  m_OpacitySlider->setMaximum(100);
  m_OpacitySlider->setSingleStep(1);
  m_OpacitySlider->setPageStep(10);

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
IGIVLVideoOverlayWidget::~IGIVLVideoOverlayWidget()
{
  this->DeRegisterDataStorageListeners();
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayWidget::SetBackgroundColour(unsigned int aabbggrr)
{
  float   r = (aabbggrr & 0xFF) / 255.0f;
  float   g = ((aabbggrr & 0xFF00) >> 8) / 255.0f;
  float   b = ((aabbggrr & 0xFF0000) >> 16) / 255.0f;
  m_LeftOverlayViewer->SetBackgroundColour(r, g, b);
  m_RightOverlayViewer->SetBackgroundColour(r, g, b);
  m_TrackedViewer->SetBackgroundColour(r, g, b);
  m_3DViewer->SetBackgroundColour(r, g, b);
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayWidget::SetOclResourceService(OclResourceService* oclserv)
{
  m_LeftOverlayViewer->SetOclResourceService(oclserv);
  m_RightOverlayViewer->SetOclResourceService(oclserv);
  m_TrackedViewer->SetOclResourceService(oclserv);
  m_3DViewer->SetOclResourceService(oclserv);
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayWidget::DeRegisterDataStorageListeners()
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->ChangedNodeEvent.RemoveListener
      (mitk::MessageDelegate1<IGIVLVideoOverlayWidget, const mitk::DataNode*>
      (this, &IGIVLVideoOverlayWidget::NodeChanged ) );
  }
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayWidget::OnLeftOverlayCheckBoxChecked(bool checked)
{
  m_LeftOverlayViewer->setVisible(checked);
  m_OverlayViewers->setVisible(   m_LeftImageCheckBox->isChecked()
                               || m_RightImageCheckBox->isChecked()
                               || m_TrackedViewCheckBox->isChecked());
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayWidget::OnRightOverlayCheckBoxChecked(bool checked)
{
  m_RightOverlayViewer->setVisible(checked);
  m_OverlayViewers->setVisible(   m_LeftImageCheckBox->isChecked()
                               || m_RightImageCheckBox->isChecked()
                               || m_TrackedViewCheckBox->isChecked());
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayWidget::On3DViewerCheckBoxChecked(bool checked)
{
  m_3DViewer->setVisible(checked);
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayWidget::OnTrackedViewerCheckBoxChecked(bool checked)
{
  m_TrackedViewer->setVisible(checked);
  m_OverlayViewers->setVisible(   m_LeftImageCheckBox->isChecked()
                               || m_RightImageCheckBox->isChecked()
                               || m_TrackedViewCheckBox->isChecked());
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayWidget::OnOpacitySliderMoved(int value)
{
  MITK_WARN << "Not implemented yet.";
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayWidget::OnLeftImageSelected(const mitk::DataNode* node)
{
  if (node != nullptr)
  {
    m_LeftOverlayViewer->SetBackgroundNode(node);
    m_TrackedViewer->SetBackgroundNode(node);
  }
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayWidget::OnRightImageSelected(const mitk::DataNode* node)
{
  if (node != nullptr)
  {
    m_RightOverlayViewer->SetBackgroundNode(node);
  }
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayWidget::OnTransformSelected(const mitk::DataNode* node)
{
  if (node != nullptr)
  {
    m_LeftOverlayViewer->SetCameraTrackingNode(node);
    m_RightOverlayViewer->SetCameraTrackingNode(node);
    m_TrackedViewer->SetCameraTrackingNode(node);
  }
}


//-----------------------------------------------------------------------------
void IGIVLVideoOverlayWidget::SetDataStorage(mitk::DataStorage* storage)
{
  if (m_DataStorage.IsNotNull() && m_DataStorage != storage)
  {
    this->DeRegisterDataStorageListeners();
  }

  m_DataStorage = storage;
  
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->ChangedNodeEvent.AddListener
      (mitk::MessageDelegate1<IGIVLVideoOverlayWidget, const mitk::DataNode*>
      (this, &IGIVLVideoOverlayWidget::NodeChanged ) );
  }

  m_3DViewer->SetDataStorage(storage);
  m_LeftOverlayViewer->SetDataStorage(storage);
  m_RightOverlayViewer->SetDataStorage(storage);
  m_TrackedViewer->SetDataStorage(storage);

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

  mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::Pointer isTransform =
      mitk::TNodePredicateDataType<mitk::CoordinateAxesData>::New();

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
void IGIVLVideoOverlayWidget::NodeChanged(const mitk::DataNode* node)
{
}

} // end namespace

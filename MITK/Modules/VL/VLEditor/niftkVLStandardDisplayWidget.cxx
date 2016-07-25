/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkVLStandardDisplayWidget.h"
#include <VLQtWidget.h>
#include <niftkSharedOGLContext.h>

namespace niftk
{

//-----------------------------------------------------------------------------
VLStandardDisplayWidget::VLStandardDisplayWidget(QWidget * /*parent*/)
{
  this->setupUi(this);

  m_GridLayout->setContentsMargins(0, 0, 0, 0);
  m_AxialViewer = new VLQtWidget(this, niftk::SharedOGLContext::GetShareWidget());
  m_AxialViewer->setObjectName("VLVideoOverlayWidget::m_AxialViewer");
  m_GridLayout->addWidget(m_AxialViewer, 0, 0);
  m_SagittalViewer = new VLQtWidget(this, niftk::SharedOGLContext::GetShareWidget());
  m_SagittalViewer->setObjectName("VLVideoOverlayWidget::m_SagittalViewer");
  m_GridLayout->addWidget(m_SagittalViewer, 0, 1);
  m_CoronalViewer = new VLQtWidget(this, niftk::SharedOGLContext::GetShareWidget());
  m_CoronalViewer->setObjectName("VLVideoOverlayWidget::m_CoronalViewer");
  m_GridLayout->addWidget(m_CoronalViewer, 1, 0);
  m_3DViewer = new VLQtWidget(this, niftk::SharedOGLContext::GetShareWidget());
  m_3DViewer->setObjectName("VLVideoOverlayWidget::m_3DViewer");
  m_GridLayout->addWidget(m_3DViewer, 1, 1);
}


//-----------------------------------------------------------------------------
VLStandardDisplayWidget::~VLStandardDisplayWidget()
{
  this->DeRegisterDataStorageListeners();
}


//-----------------------------------------------------------------------------
void VLStandardDisplayWidget::SetBackgroundColour(unsigned int aabbggrr)
{
  float   r = (aabbggrr & 0xFF) / 255.0f;
  float   g = ((aabbggrr & 0xFF00) >> 8) / 255.0f;
  float   b = ((aabbggrr & 0xFF0000) >> 16) / 255.0f;
  m_AxialViewer->vlSceneView()->setBackgroundColour(r, g, b);
  m_SagittalViewer->vlSceneView()->setBackgroundColour(r, g, b);
  m_CoronalViewer->vlSceneView()->setBackgroundColour(r, g, b);
  m_3DViewer->vlSceneView()->setBackgroundColour(r, g, b);
}


//-----------------------------------------------------------------------------
void VLStandardDisplayWidget::DeRegisterDataStorageListeners()
{
  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->ChangedNodeEvent.RemoveListener
      (mitk::MessageDelegate1<VLStandardDisplayWidget, const mitk::DataNode*>
      (this, &VLStandardDisplayWidget::NodeChanged ) );
  }
}


//-----------------------------------------------------------------------------
void VLStandardDisplayWidget::SetDataStorage(mitk::DataStorage* storage)
{
  if (m_DataStorage.IsNotNull() && m_DataStorage != storage)
  {
    this->DeRegisterDataStorageListeners();
  }

  m_DataStorage = storage;

  if (m_DataStorage.IsNotNull())
  {
    m_DataStorage->ChangedNodeEvent.AddListener
      (mitk::MessageDelegate1<VLStandardDisplayWidget, const mitk::DataNode*>
      (this, &VLStandardDisplayWidget::NodeChanged ) );
  }

  m_AxialViewer->vlSceneView()->setDataStorage( storage );
  m_SagittalViewer->vlSceneView()->setDataStorage( storage );
  m_CoronalViewer->vlSceneView()->setDataStorage( storage );
  m_3DViewer->vlSceneView()->setDataStorage( storage );
}


//-----------------------------------------------------------------------------
void VLStandardDisplayWidget::NodeChanged(const mitk::DataNode* node)
{
}

} // end namespace

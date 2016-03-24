/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseSegmentorWidget.h"

#include <mitkToolManager.h>

#include <niftkSegmentationSelectorWidget.h>
#include <niftkToolSelectorWidget.h>


//-----------------------------------------------------------------------------
niftkBaseSegmentorWidget::niftkBaseSegmentorWidget(QWidget* parent)
  : m_SegmentationSelectorWidget(nullptr),
    m_ToolSelectorWidget(nullptr),
    m_ContainerForSelectorWidget(nullptr),
    m_ContainerForToolWidget(nullptr)
{
  // Set up the Image and Segmentation Selector.
  // Subclasses add it to their layouts, at the appropriate point.
  m_ContainerForSelectorWidget = new QWidget(parent);
  m_SegmentationSelectorWidget = new niftkSegmentationSelectorWidget(m_ContainerForSelectorWidget);
  m_SegmentationSelectorWidget->m_NewSegmentationButton->setEnabled(false);
  m_SegmentationSelectorWidget->m_ReferenceImageNameLabel->setText("<font color='red'>&lt;not selected&gt;</font>");
  m_SegmentationSelectorWidget->m_ReferenceImageNameLabel->show();
  m_SegmentationSelectorWidget->m_SegmentationImageNameLabel->setText("<font color='red'>&lt;not selected&gt;</font>");
  m_SegmentationSelectorWidget->m_SegmentationImageNameLabel->show();

  // Set up the Tool Selector.
  // Subclasses add it to their layouts, at the appropriate point.
  m_ContainerForToolWidget = new QWidget(parent);
  m_ToolSelectorWidget = new niftkToolSelectorWidget(m_ContainerForToolWidget);
  m_ToolSelectorWidget->m_ManualToolSelectionBox->SetGenerateAccelerators(true);
  m_ToolSelectorWidget->m_ManualToolSelectionBox->SetLayoutColumns(3);
  m_ToolSelectorWidget->m_ManualToolSelectionBox->SetToolGUIArea(m_ToolSelectorWidget->m_ManualToolGUIContainer);
  m_ToolSelectorWidget->m_ManualToolSelectionBox->SetEnabledMode(QmitkToolSelectionBox::EnabledWithWorkingData);
}


//-----------------------------------------------------------------------------
niftkBaseSegmentorWidget::~niftkBaseSegmentorWidget()
{
  if (m_SegmentationSelectorWidget != NULL)
  {
    delete m_SegmentationSelectorWidget;
  }

  if (m_ToolSelectorWidget != NULL)
  {
    delete m_ToolSelectorWidget;
  }
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorWidget::CreateConnections()
{
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorWidget::EnableSegmentationWidgets(bool enabled)
{
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorWidget::SetEnableManualToolSelectionBox(bool enabled)
{
  m_ToolSelectorWidget->m_ManualToolSelectionBox->QWidget::setEnabled(enabled);
  m_ToolSelectorWidget->m_ManualToolGUIContainer->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorWidget::SetToolManager(mitk::ToolManager* toolManager)
{
  m_ToolSelectorWidget->SetToolManager(toolManager);
}


//-----------------------------------------------------------------------------
mitk::ToolManager* niftkBaseSegmentorWidget::GetToolManager() const
{
  return m_ToolSelectorWidget->GetToolManager();
}

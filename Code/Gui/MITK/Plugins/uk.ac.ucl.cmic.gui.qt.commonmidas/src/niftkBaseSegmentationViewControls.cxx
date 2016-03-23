/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseSegmentationViewControls.h"

#include <berryPlatform.h>

#include <QMessageBox>
#include "internal/MIDASActivator.h"
#include <mitkILinkedRenderWindowPart.h>
#include <mitkImageAccessByItk.h>
#include <mitkDataNodeObject.h>
#include <mitkProperties.h>
#include <mitkColorProperty.h>
#include <mitkRenderingManager.h>
#include <mitkBaseRenderer.h>
#include <mitkSegTool2D.h>
#include <mitkVtkResliceInterpolationProperty.h>
#include <mitkToolManager.h>
#include <mitkGlobalInteraction.h>
#include <mitkStateMachine.h>
#include <mitkDataStorageUtils.h>
#include <mitkColorProperty.h>
#include <mitkProperties.h>
#include <QmitkRenderWindow.h>

#include <NifTKConfigure.h>
#include <niftkMIDASNewSegmentationDialog.h>
#include <niftkMIDASTool.h>
#include <niftkMIDASDrawTool.h>
#include <niftkMIDASPolyTool.h>
#include <niftkMIDASSeedTool.h>
#include <niftkMIDASOrientationUtils.h>

//-----------------------------------------------------------------------------
niftkBaseSegmentationViewControls::niftkBaseSegmentationViewControls(QWidget* parent)
  : m_SegmentationSelectorWidget(nullptr),
    m_ToolSelector(nullptr),
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
  m_ToolSelector = new niftkMIDASToolSelectorWidget(m_ContainerForToolWidget);
  m_ToolSelector->m_ManualToolSelectionBox->SetGenerateAccelerators(true);
  m_ToolSelector->m_ManualToolSelectionBox->SetLayoutColumns(3);
  m_ToolSelector->m_ManualToolSelectionBox->SetToolGUIArea(m_ToolSelector->m_ManualToolGUIContainer);
  m_ToolSelector->m_ManualToolSelectionBox->SetEnabledMode(QmitkToolSelectionBox::EnabledWithWorkingData);
}


//-----------------------------------------------------------------------------
niftkBaseSegmentationViewControls::~niftkBaseSegmentationViewControls()
{
  if (m_SegmentationSelectorWidget != NULL)
  {
    delete m_SegmentationSelectorWidget;
  }

  if (m_ToolSelector != NULL)
  {
    delete m_ToolSelector;
  }
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentationViewControls::CreateConnections()
{
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentationViewControls::EnableSegmentationWidgets(bool enabled)
{
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentationViewControls::SetEnableManualToolSelectionBox(bool enabled)
{
  m_ToolSelector->m_ManualToolSelectionBox->QWidget::setEnabled(enabled);
  m_ToolSelector->m_ManualToolGUIContainer->setEnabled(enabled);
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentationViewControls::SetToolManager(mitk::ToolManager* toolManager)
{
  m_ToolSelector->SetToolManager(toolManager);
}


//-----------------------------------------------------------------------------
mitk::ToolManager* niftkBaseSegmentationViewControls::GetToolManager() const
{
  return m_ToolSelector->GetToolManager();
}

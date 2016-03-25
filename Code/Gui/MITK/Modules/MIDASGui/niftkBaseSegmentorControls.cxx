/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseSegmentorControls.h"

#include <mitkToolManager.h>

#include <niftkSegmentationSelectorWidget.h>
#include <niftkToolSelectorWidget.h>


//-----------------------------------------------------------------------------
niftkBaseSegmentorControls::niftkBaseSegmentorControls(QWidget* parent)
  : m_SegmentationSelectorWidget(nullptr),
    m_ToolSelectorWidget(nullptr),
    m_ContainerForSelectorWidget(nullptr),
    m_ContainerForToolWidget(nullptr)
{
  // Set up the Image and Segmentation Selector.
  // Subclasses add it to their layouts, at the appropriate point.
  m_ContainerForSelectorWidget = new QWidget(parent);
  m_SegmentationSelectorWidget = new niftkSegmentationSelectorWidget(m_ContainerForSelectorWidget);

  // Set up the Tool Selector.
  // Subclasses add it to their layouts, at the appropriate point.
  m_ContainerForToolWidget = new QWidget(parent);
  m_ToolSelectorWidget = new niftkToolSelectorWidget(m_ContainerForToolWidget);

  this->connect(m_SegmentationSelectorWidget, SIGNAL(NewSegmentationButtonClicked()), SIGNAL(NewSegmentationButtonClicked()));
  this->connect(m_ToolSelectorWidget, SIGNAL(ToolSelected(int)), SIGNAL(ToolSelected(int)));
}


//-----------------------------------------------------------------------------
niftkBaseSegmentorControls::~niftkBaseSegmentorControls()
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
void niftkBaseSegmentorControls::EnableSegmentationWidgets(bool enabled)
{
}


//-----------------------------------------------------------------------------
bool niftkBaseSegmentorControls::IsToolSelectorEnabled() const
{
  return m_ToolSelectorWidget->IsEnabled();
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorControls::SetToolSelectorEnabled(bool enabled)
{
  m_ToolSelectorWidget->SetEnabled(enabled);
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorControls::SetToolManager(mitk::ToolManager* toolManager)
{
  m_ToolSelectorWidget->SetToolManager(toolManager);
}


//-----------------------------------------------------------------------------
mitk::ToolManager* niftkBaseSegmentorControls::GetToolManager() const
{
  return m_ToolSelectorWidget->GetToolManager();
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorControls::SelectReferenceImage(const QString& imageName)
{
  m_SegmentationSelectorWidget->SelectReferenceImage(imageName);
}


//-----------------------------------------------------------------------------
void niftkBaseSegmentorControls::SelectSegmentationImage(const QString& imageName)
{
  m_SegmentationSelectorWidget->SelectSegmentationImage(imageName);
}

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseSegmentorGUI.h"

#include <mitkToolManager.h>

#include "niftkSegmentationSelectorWidget.h"
#include "niftkToolSelectorWidget.h"

namespace niftk
{

//-----------------------------------------------------------------------------
BaseSegmentorGUI::BaseSegmentorGUI(QWidget* parent)
  : BaseGUI(parent),
    m_SegmentationSelectorWidget(nullptr),
    m_ToolSelectorWidget(nullptr),
    m_ContainerForSelectorWidget(nullptr),
    m_ContainerForToolWidget(nullptr),
    m_ToolManager(nullptr)
{
  // Set up the Image and Segmentation Selector.
  // Subclasses add it to their layouts, at the appropriate point.
  m_ContainerForSelectorWidget = new QWidget(parent);
  m_SegmentationSelectorWidget = new SegmentationSelectorWidget(m_ContainerForSelectorWidget);

  // Set up the Tool Selector.
  // Subclasses add it to their layouts, at the appropriate point.
  m_ContainerForToolWidget = new QWidget(parent);
  m_ToolSelectorWidget = new ToolSelectorWidget(m_ContainerForToolWidget);

  this->connect(m_SegmentationSelectorWidget, SIGNAL(NewSegmentationButtonClicked()), SIGNAL(NewSegmentationButtonClicked()));
}


//-----------------------------------------------------------------------------
BaseSegmentorGUI::~BaseSegmentorGUI()
{
  if (m_ToolManager)
  {
    m_ToolManager->WorkingDataChanged -= mitk::MessageDelegate<BaseSegmentorGUI>(this, &BaseSegmentorGUI::OnWorkingDataChanged);
  }

  if (m_SegmentationSelectorWidget)
  {
    delete m_SegmentationSelectorWidget;
  }

  if (m_ToolSelectorWidget)
  {
    delete m_ToolSelectorWidget;
  }
}


//-----------------------------------------------------------------------------
void BaseSegmentorGUI::EnableSegmentationWidgets(bool enabled)
{
}


//-----------------------------------------------------------------------------
bool BaseSegmentorGUI::IsToolSelectorEnabled() const
{
  return m_ToolSelectorWidget->IsEnabled();
}


//-----------------------------------------------------------------------------
void BaseSegmentorGUI::SetToolSelectorEnabled(bool enabled)
{
  m_ToolSelectorWidget->SetEnabled(enabled);
}


//-----------------------------------------------------------------------------
void BaseSegmentorGUI::SetToolManager(mitk::ToolManager* toolManager)
{
  if (toolManager != m_ToolManager)
  {
    if (m_ToolManager)
    {
      m_ToolManager->WorkingDataChanged -= mitk::MessageDelegate<BaseSegmentorGUI>(this, &BaseSegmentorGUI::OnWorkingDataChanged);
    }

    if (toolManager)
    {
      toolManager->WorkingDataChanged += mitk::MessageDelegate<BaseSegmentorGUI>(this, &BaseSegmentorGUI::OnWorkingDataChanged);
    }

    m_ToolManager = toolManager;

    m_ToolSelectorWidget->SetToolManager(toolManager);
    m_SegmentationSelectorWidget->SetToolManager(toolManager);

    this->OnWorkingDataChanged();
  }
}


//-----------------------------------------------------------------------------
mitk::ToolManager* BaseSegmentorGUI::GetToolManager() const
{
  return m_ToolManager;
}


//-----------------------------------------------------------------------------
void BaseSegmentorGUI::OnWorkingDataChanged()
{
  bool hasWorkingData = m_ToolManager && !m_ToolManager->GetWorkingData().empty();

  this->EnableSegmentationWidgets(hasWorkingData);
}

}

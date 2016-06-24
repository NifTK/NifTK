/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkToolSelectorWidget.h"
#include <mitkToolManagerProvider.h>

namespace niftk
{

//-----------------------------------------------------------------------------
ToolSelectorWidget::ToolSelectorWidget(QWidget *parent)
{
  this->setupUi(parent);

  /// Note: If we did not set a layout for the GUI container then one would be
  /// created with 9,9,9,9 margins.
  m_ManualToolGUIContainer->setLayout(new QHBoxLayout());
  m_ManualToolGUIContainer->layout()->setContentsMargins(6, 0, 6, 0);

  /// Note:
  /// We set a minimum height for the tool GUI container so that if a tool
  /// needs additional GUI controls, they can be put into that 'empty' area,
  /// and the rest of the GUI controls do not 'jump down'. From the MIDAS
  /// tools only the paintbrush tool (morphological editor) and the draw tool
  /// (irregular editor) have such controls, both provide one slider.
  /// On Mac OS X Mountain Lion the required height for this is 24 pixels,
  /// but this might be different on other platforms.
  m_ManualToolGUIContainer->setMinimumHeight(24);

  m_ManualToolSelectionBox->SetGenerateAccelerators(true);
  m_ManualToolSelectionBox->SetLayoutColumns(3);
  m_ManualToolSelectionBox->SetToolGUIArea(m_ManualToolGUIContainer);
  m_ManualToolSelectionBox->SetEnabledMode(QmitkToolSelectionBox::EnabledWithWorkingData);
}


//-----------------------------------------------------------------------------
ToolSelectorWidget::~ToolSelectorWidget()
{
}


//-----------------------------------------------------------------------------
mitk::ToolManager* ToolSelectorWidget::GetToolManager() const
{
  return m_ManualToolSelectionBox->GetToolManager();
}


//-----------------------------------------------------------------------------
void ToolSelectorWidget::SetToolManager(mitk::ToolManager* toolManager)
{
  m_ManualToolSelectionBox->SetToolManager(*toolManager);
}


//-----------------------------------------------------------------------------
void ToolSelectorWidget::SetEnabled(bool enabled)
{
  this->setEnabled(enabled);
  m_ManualToolSelectionBox->QWidget::setEnabled(enabled);
  m_ManualToolGUIContainer->setEnabled(enabled);

  if (!enabled)
  {
    int activeToolID = m_ManualToolSelectionBox->GetToolManager()->GetActiveToolID();
    if (activeToolID != -1)
    {
      m_ManualToolSelectionBox->GetToolManager()->ActivateTool(-1);
    }
  }
}


//-----------------------------------------------------------------------------
bool ToolSelectorWidget::IsEnabled() const
{
  return this->isEnabled();
}


//-----------------------------------------------------------------------------
void ToolSelectorWidget::SetDisplayedToolGroups(const QString& toolGroups)
{
  m_ManualToolSelectionBox->SetDisplayedToolGroups(toolGroups.toStdString());
}

}

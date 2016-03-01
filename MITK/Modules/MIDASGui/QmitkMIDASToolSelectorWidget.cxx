/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMIDASToolSelectorWidget.h"
#include <mitkToolManagerProvider.h>

//-----------------------------------------------------------------------------
QmitkMIDASToolSelectorWidget::QmitkMIDASToolSelectorWidget(QWidget *parent)
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

  this->connect(m_ManualToolSelectionBox, SIGNAL(ToolSelected(int)), SLOT(OnToolSelected(int)));
}


//-----------------------------------------------------------------------------
QmitkMIDASToolSelectorWidget::~QmitkMIDASToolSelectorWidget()
{
}


//-----------------------------------------------------------------------------
mitk::ToolManager* QmitkMIDASToolSelectorWidget::GetToolManager() const
{
  return m_ManualToolSelectionBox->GetToolManager();
}

//-----------------------------------------------------------------------------
void QmitkMIDASToolSelectorWidget::SetToolManager(mitk::ToolManager& toolManager) // no NULL pointer allowed here, a manager is required
{
  m_ManualToolSelectionBox->SetToolManager(toolManager);
}


//-----------------------------------------------------------------------------
void QmitkMIDASToolSelectorWidget::SetEnabled(bool enabled)
{
  this->setEnabled(enabled);
  m_ManualToolSelectionBox->QWidget::setEnabled(enabled);
  m_ManualToolGUIContainer->setEnabled(enabled);

  if (!enabled)
  {
    int activeToolID = this->GetActiveToolID();
    if (activeToolID != -1)
    {
      m_ManualToolSelectionBox->GetToolManager()->ActivateTool(-1);
    }
  }
}


//-----------------------------------------------------------------------------
bool QmitkMIDASToolSelectorWidget::GetEnabled() const
{
  return this->isEnabled();
}


//-----------------------------------------------------------------------------
int QmitkMIDASToolSelectorWidget::GetActiveToolID()
{
  return m_ManualToolSelectionBox->GetToolManager()->GetActiveToolID();
}


//-----------------------------------------------------------------------------
void QmitkMIDASToolSelectorWidget::OnToolSelected(int toolId)
{
  emit ToolSelected(toolId);
}

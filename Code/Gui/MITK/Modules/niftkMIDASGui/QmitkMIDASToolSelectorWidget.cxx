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
  setupUi(parent);
  connect(m_ManualToolSelectionBox, SIGNAL(ToolSelected(int)), this, SLOT(OnToolSelected(int)));
}


//-----------------------------------------------------------------------------
QmitkMIDASToolSelectorWidget::~QmitkMIDASToolSelectorWidget()
{

}


//-----------------------------------------------------------------------------
mitk::ToolManager* QmitkMIDASToolSelectorWidget::GetToolManager() const
{
  return mitk::ToolManagerProvider::GetInstance()->GetToolManager();
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

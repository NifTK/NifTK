/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkMIDASToolSelectorWidget.h"

QmitkMIDASToolSelectorWidget::QmitkMIDASToolSelectorWidget(QWidget *parent)
{
  setupUi(parent);
  connect(m_ManualToolSelectionBox, SIGNAL(ToolSelected(int)), this, SLOT(OnToolSelected(int)));
}

QmitkMIDASToolSelectorWidget::~QmitkMIDASToolSelectorWidget()
{

}

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

bool QmitkMIDASToolSelectorWidget::GetEnabled() const
{
  return this->isEnabled();
}

int QmitkMIDASToolSelectorWidget::GetActiveToolID()
{
  return m_ManualToolSelectionBox->GetToolManager()->GetActiveToolID();
}

void QmitkMIDASToolSelectorWidget::OnToolSelected(int toolId)
{
  emit ToolSelected(toolId);
}

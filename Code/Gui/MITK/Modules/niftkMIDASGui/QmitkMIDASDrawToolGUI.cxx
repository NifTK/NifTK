/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMIDASDrawToolGUI.h"

#include <ctkSliderWidget.h>

#include <QLabel>
#include <QLayout>
#include <QPainter>


MITK_TOOL_GUI_MACRO(NIFTKMIDASGUI_EXPORT, QmitkMIDASDrawToolGUI, "")


//-----------------------------------------------------------------------------
QmitkMIDASDrawToolGUI::QmitkMIDASDrawToolGUI()
:QmitkToolGUI()
, m_Slider(NULL)
, m_Frame(NULL)
{
  // create the visible widgets
  QBoxLayout* layout = new QHBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->setSpacing(3);

  QLabel* label = new QLabel("Eraser radius (mm):", this);
  layout->addWidget(label);

  m_Slider = new ctkSliderWidget(this);
  m_Slider->layout()->setSpacing(3);
  m_Slider->setMinimum(0.5);
  m_Slider->setMaximum(30.0);
  m_Slider->setSingleStep(0.5);
  m_Slider->setPageStep(0.1);
  m_Slider->setValue(0.5);
  layout->addWidget(m_Slider);

  this->connect(m_Slider, SIGNAL(valueChanged(double)), SLOT(OnSliderValueChanged(double)));
  this->connect(this, SIGNAL(NewToolAssociated(mitk::Tool*)), SLOT(OnNewToolAssociated(mitk::Tool*)));
}


//-----------------------------------------------------------------------------
QmitkMIDASDrawToolGUI::~QmitkMIDASDrawToolGUI()
{
  if (m_DrawTool.IsNotNull())
  {
    m_DrawTool->CursorSizeChanged -= mitk::MessageDelegate1<QmitkMIDASDrawToolGUI, double>(this, &QmitkMIDASDrawToolGUI::OnCursorSizeChanged);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASDrawToolGUI::OnNewToolAssociated(mitk::Tool* tool)
{
  if (m_DrawTool.IsNotNull())
  {
    m_DrawTool->CursorSizeChanged -= mitk::MessageDelegate1<QmitkMIDASDrawToolGUI, double>(this, &QmitkMIDASDrawToolGUI::OnCursorSizeChanged);
  }

  m_DrawTool = dynamic_cast<mitk::MIDASDrawTool*>(tool);

  if (m_DrawTool.IsNotNull())
  {
    this->OnCursorSizeChanged(m_DrawTool->GetCursorSize());
    m_DrawTool->CursorSizeChanged += mitk::MessageDelegate1<QmitkMIDASDrawToolGUI, double>(this, &QmitkMIDASDrawToolGUI::OnCursorSizeChanged);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASDrawToolGUI::OnSliderValueChanged(double value)
{
  if (m_DrawTool.IsNotNull())
  {
    m_DrawTool->SetCursorSize(value);
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASDrawToolGUI::OnCursorSizeChanged(double cursorSize)
{
  m_Slider->setValue(cursorSize);
}

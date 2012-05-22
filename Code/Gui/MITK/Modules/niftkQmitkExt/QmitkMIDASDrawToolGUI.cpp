/*=========================================================================

Program:   Medical Imaging & Interaction Toolkit
Language:  C++
Date:      $Date: 2012-02-16 21:02:48 +0000 (Thu, 16 Feb 2012) $
Version:   $Revision: 8525 $

Copyright (c) German Cancer Research Center, Division of Medical and
Biological Informatics. All rights reserved.
See MITKCopyright.txt or http://www.mitk.org/copyright.html for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#include "QmitkMIDASDrawToolGUI.h"

#include <qlabel.h>
#include <qslider.h>
#include <qpushbutton.h>
#include <qlayout.h>
#include <qpainter.h>

MITK_TOOL_GUI_MACRO(NIFTKQMITKEXT_EXPORT, QmitkMIDASDrawToolGUI, "")

QmitkMIDASDrawToolGUI::QmitkMIDASDrawToolGUI()
:QmitkToolGUI()
, m_Slider(NULL)
, m_SizeLabel(NULL)
, m_Frame(NULL)
{
  // create the visible widgets
  QBoxLayout* layout = new QHBoxLayout( this );
  this->setContentsMargins( 0, 0, 0, 0 );

  QLabel* label = new QLabel( "Eraser ", this );
  layout->addWidget(label);

  m_SizeLabel = new QLabel( " 1", this );
  layout->addWidget(m_SizeLabel);

  m_Slider = new QSlider( Qt::Horizontal, this );
  m_Slider->setMinimum(1);
  m_Slider->setMaximum(30);
  m_Slider->setPageStep(1);
  m_Slider->setValue(1);
  connect( m_Slider, SIGNAL(valueChanged(int)), this, SLOT(OnSliderValueChanged(int)));
  layout->addWidget( m_Slider );

  connect( this, SIGNAL(NewToolAssociated(mitk::Tool*)), this, SLOT(OnNewToolAssociated(mitk::Tool*)) );
}

QmitkMIDASDrawToolGUI::~QmitkMIDASDrawToolGUI()
{
  if (m_DrawTool.IsNotNull())
  {
    m_DrawTool->CursorSizeChanged -= mitk::MessageDelegate1<QmitkMIDASDrawToolGUI, int>( this, &QmitkMIDASDrawToolGUI::OnCursorSizeChanged );
  }
}

void QmitkMIDASDrawToolGUI::OnNewToolAssociated(mitk::Tool* tool)
{
  if (m_DrawTool.IsNotNull())
  {
    m_DrawTool->CursorSizeChanged -= mitk::MessageDelegate1<QmitkMIDASDrawToolGUI, int>( this, &QmitkMIDASDrawToolGUI::OnCursorSizeChanged );
  }

  m_DrawTool = dynamic_cast<mitk::MIDASDrawTool*>( tool );

  if (m_DrawTool.IsNotNull())
  {
    m_DrawTool->CursorSizeChanged += mitk::MessageDelegate1<QmitkMIDASDrawToolGUI, int>( this, &QmitkMIDASDrawToolGUI::OnCursorSizeChanged );
  }
}

void QmitkMIDASDrawToolGUI::OnSliderValueChanged(int value)
{
  if (m_DrawTool.IsNotNull())
  {
    m_DrawTool->SetCursorSize( value );
    m_SizeLabel->setText(QString::number(value));
  }
}

void QmitkMIDASDrawToolGUI::OnCursorSizeChanged(int current)
{
  m_Slider->setValue(current);
}


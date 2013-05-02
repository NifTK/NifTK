/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkMIDASPaintbrushToolGUI.h"

#include <qlabel.h>
#include <qslider.h>
#include <qpushbutton.h>
#include <qlayout.h>
#include <qpainter.h>

MITK_TOOL_GUI_MACRO(NIFTKMIDASGUI_EXPORT, QmitkMIDASPaintbrushToolGUI, "")


//-----------------------------------------------------------------------------
QmitkMIDASPaintbrushToolGUI::QmitkMIDASPaintbrushToolGUI()
:QmitkToolGUI()
, m_Slider(NULL)
, m_SizeLabel(NULL)
, m_Frame(NULL)
{
  // create the visible widgets
  QBoxLayout* layout = new QHBoxLayout( this );
  this->setContentsMargins( 0, 0, 0, 0 );

  QLabel* label = new QLabel( "Cursor Width ", this );
  layout->addWidget(label);

  m_SizeLabel = new QLabel( " 1", this );
  layout->addWidget(m_SizeLabel);

  m_Slider = new QSlider( Qt::Horizontal, this );
  m_Slider->setMinimum(1);
  m_Slider->setMaximum(6);
  m_Slider->setPageStep(1);
  m_Slider->setValue(1);
  connect( m_Slider, SIGNAL(valueChanged(int)), this, SLOT(OnSliderValueChanged(int)));
  layout->addWidget( m_Slider );

  connect( this, SIGNAL(NewToolAssociated(mitk::Tool*)), this, SLOT(OnNewToolAssociated(mitk::Tool*)) );
}


//-----------------------------------------------------------------------------
QmitkMIDASPaintbrushToolGUI::~QmitkMIDASPaintbrushToolGUI()
{
  if (m_PaintbrushTool.IsNotNull())
  {
    m_PaintbrushTool->CursorSizeChanged -= mitk::MessageDelegate1<QmitkMIDASPaintbrushToolGUI, int>( this, &QmitkMIDASPaintbrushToolGUI::OnCursorSizeChanged );
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASPaintbrushToolGUI::OnNewToolAssociated(mitk::Tool* tool)
{
  if (m_PaintbrushTool.IsNotNull())
  {
    m_PaintbrushTool->CursorSizeChanged -= mitk::MessageDelegate1<QmitkMIDASPaintbrushToolGUI, int>( this, &QmitkMIDASPaintbrushToolGUI::OnCursorSizeChanged );
  }

  m_PaintbrushTool = dynamic_cast<mitk::MIDASPaintbrushTool*>( tool );

  if (m_PaintbrushTool.IsNotNull())
  {
    m_PaintbrushTool->CursorSizeChanged += mitk::MessageDelegate1<QmitkMIDASPaintbrushToolGUI, int>( this, &QmitkMIDASPaintbrushToolGUI::OnCursorSizeChanged );
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASPaintbrushToolGUI::OnSliderValueChanged(int value)
{
  if (m_PaintbrushTool.IsNotNull())
  {
    m_PaintbrushTool->SetCursorSize( value );
    m_SizeLabel->setText(QString::number(value));
  }
}


//-----------------------------------------------------------------------------
void QmitkMIDASPaintbrushToolGUI::OnCursorSizeChanged(int current)
{
  m_Slider->setValue(current);
}


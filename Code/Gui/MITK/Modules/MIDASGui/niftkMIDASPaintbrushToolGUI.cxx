/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMIDASPaintbrushToolGUI.h"

#include <qlabel.h>
#include <qslider.h>
#include <qpushbutton.h>
#include <qlayout.h>
#include <qpainter.h>

#include <niftkToolFactoryMacros.h>


NIFTK_TOOL_GUI_MACRO(NIFTKMIDASGUI_EXPORT, MIDASPaintbrushTool, niftkMIDASPaintbrushToolGUI, "MIDAS Paintbrush Tool GUI")

//-----------------------------------------------------------------------------
niftkMIDASPaintbrushToolGUI::niftkMIDASPaintbrushToolGUI()
:QmitkToolGUI()
, m_Slider(NULL)
, m_SizeLabel(NULL)
, m_Frame(NULL)
{
  // create the visible widgets
  QBoxLayout* layout = new QHBoxLayout( this );
  layout->setContentsMargins(0, 0, 0, 0);
  layout->setSpacing(3);
  this->setLayout(layout);

  QLabel* label = new QLabel( "cursor width:", this );
  layout->addWidget(label);

  m_SizeLabel = new QLabel("1", this);
  layout->addWidget(m_SizeLabel);

  m_Slider = new QSlider(Qt::Horizontal, this);
  m_Slider->setMinimum(1);
  m_Slider->setMaximum(6);
  m_Slider->setPageStep(1);
  m_Slider->setValue(1);
  this->connect(m_Slider, SIGNAL(valueChanged(int)), SLOT(OnSliderValueChanged(int)));
  layout->addWidget(m_Slider);

  this->connect(this, SIGNAL(NewToolAssociated(mitk::Tool*)), SLOT(OnNewToolAssociated(mitk::Tool*)));
}


//-----------------------------------------------------------------------------
niftkMIDASPaintbrushToolGUI::~niftkMIDASPaintbrushToolGUI()
{
  if (m_PaintbrushTool.IsNotNull())
  {
    m_PaintbrushTool->CursorSizeChanged -= mitk::MessageDelegate1<niftkMIDASPaintbrushToolGUI, int>( this, &niftkMIDASPaintbrushToolGUI::OnCursorSizeChanged );
  }
}


//-----------------------------------------------------------------------------
void niftkMIDASPaintbrushToolGUI::OnNewToolAssociated(mitk::Tool* tool)
{
  if (m_PaintbrushTool.IsNotNull())
  {
    m_PaintbrushTool->CursorSizeChanged -= mitk::MessageDelegate1<niftkMIDASPaintbrushToolGUI, int>( this, &niftkMIDASPaintbrushToolGUI::OnCursorSizeChanged );
  }

  m_PaintbrushTool = dynamic_cast<niftk::MIDASPaintbrushTool*>( tool );

  if (m_PaintbrushTool.IsNotNull())
  {
    m_PaintbrushTool->CursorSizeChanged += mitk::MessageDelegate1<niftkMIDASPaintbrushToolGUI, int>( this, &niftkMIDASPaintbrushToolGUI::OnCursorSizeChanged );
  }
}


//-----------------------------------------------------------------------------
void niftkMIDASPaintbrushToolGUI::OnSliderValueChanged(int value)
{
  if (m_PaintbrushTool.IsNotNull())
  {
    m_PaintbrushTool->SetCursorSize( value );
    m_SizeLabel->setText(QString::number(value));
  }
}


//-----------------------------------------------------------------------------
void niftkMIDASPaintbrushToolGUI::OnCursorSizeChanged(int current)
{
  m_Slider->setValue(current);
}


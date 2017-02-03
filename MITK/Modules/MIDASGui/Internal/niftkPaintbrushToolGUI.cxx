/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPaintbrushToolGUI.h"

#include <qlabel.h>
#include <qslider.h>
#include <qpushbutton.h>
#include <qlayout.h>
#include <qpainter.h>

#include <niftkToolFactoryMacros.h>

namespace niftk
{

NIFTK_TOOL_GUI_MACRO_NO_EXPORT(PaintbrushTool, PaintbrushToolGUI, "Paintbrush Tool GUI")

//-----------------------------------------------------------------------------
PaintbrushToolGUI::PaintbrushToolGUI()
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

  QLabel* label = new QLabel( "eraser width (voxel):", this );
  layout->addWidget(label);

  m_SizeLabel = new QLabel("1", this);
  layout->addWidget(m_SizeLabel);

  m_Slider = new QSlider(Qt::Horizontal, this);
  m_Slider->setMinimum(1);
  m_Slider->setMaximum(6);
  m_Slider->setPageStep(1);
  m_Slider->setValue(1);
  this->connect(m_Slider, SIGNAL(valueChanged(int)), SLOT(OnEraserSizeChangedInGui(int)));
  layout->addWidget(m_Slider);

  this->connect(this, SIGNAL(NewToolAssociated(mitk::Tool*)), SLOT(OnNewToolAssociated(mitk::Tool*)));
}


//-----------------------------------------------------------------------------
PaintbrushToolGUI::~PaintbrushToolGUI()
{
  if (m_PaintbrushTool.IsNotNull())
  {
    m_PaintbrushTool->EraserSizeChanged -= mitk::MessageDelegate1<PaintbrushToolGUI, int>( this, &PaintbrushToolGUI::OnEraserSizeChangedInTool );
  }
}


//-----------------------------------------------------------------------------
void PaintbrushToolGUI::OnNewToolAssociated(mitk::Tool* tool)
{
  if (m_PaintbrushTool.IsNotNull())
  {
    m_PaintbrushTool->EraserSizeChanged -= mitk::MessageDelegate1<PaintbrushToolGUI, int>( this, &PaintbrushToolGUI::OnEraserSizeChangedInTool );
  }

  m_PaintbrushTool = dynamic_cast<PaintbrushTool*>( tool );

  if (m_PaintbrushTool.IsNotNull())
  {
    m_PaintbrushTool->EraserSizeChanged += mitk::MessageDelegate1<PaintbrushToolGUI, int>( this, &PaintbrushToolGUI::OnEraserSizeChangedInTool );
  }
}


//-----------------------------------------------------------------------------
void PaintbrushToolGUI::OnEraserSizeChangedInGui(int value)
{
  if (m_PaintbrushTool.IsNotNull())
  {
    m_PaintbrushTool->SetEraserSize( value );
    m_SizeLabel->setText(QString::number(value));
  }
}


//-----------------------------------------------------------------------------
void PaintbrushToolGUI::OnEraserSizeChangedInTool(int current)
{
  m_Slider->setValue(current);
}

}

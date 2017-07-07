/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPaintbrushToolGUI.h"

#include <QLabel>
#include <QLayout>
#include <QPainter>
#include <QPushButton>
#include <QTimer>

#include <ctkSliderWidget.h>

#include <mitkFocusManager.h>
#include <mitkGlobalInteraction.h>

#include <niftkToolFactoryMacros.h>

namespace niftk
{

NIFTK_TOOL_GUI_MACRO_NO_EXPORT(PaintbrushTool, PaintbrushToolGUI, "Paintbrush Tool GUI")

//-----------------------------------------------------------------------------
PaintbrushToolGUI::PaintbrushToolGUI()
  : QmitkToolGUI(),
    m_Slider(nullptr),
    m_Frame(nullptr),
    m_ShowEraserTimer(new QTimer(this))
{
  // create the visible widgets
  QBoxLayout* layout = new QHBoxLayout( this );
  layout->setContentsMargins(0, 0, 0, 0);
  layout->setSpacing(3);
  this->setLayout(layout);

  QLabel* label = new QLabel( "Eraser width (voxel):", this );
  layout->addWidget(label);

  m_Slider = new ctkSliderWidget(this);
  m_Slider->layout()->setSpacing(3);
  m_Slider->setMinimum(1.0);
  m_Slider->setMaximum(25.0);
  m_Slider->setSingleStep(2.0);
  m_Slider->setPageStep(2.0);
  m_Slider->setValue(1.0);
  m_Slider->setDecimals(0);
  layout->addWidget(m_Slider);

  this->connect(m_Slider, SIGNAL(valueChanged(double)), SLOT(OnEraserSizeChangedInGui(double)));
  this->connect(this, SIGNAL(NewToolAssociated(mitk::Tool*)), SLOT(OnNewToolAssociated(mitk::Tool*)));

  m_ShowEraserTimer->setInterval(1000);
  m_ShowEraserTimer->setSingleShot(true);

  this->connect(m_ShowEraserTimer, SIGNAL(timeout()), SLOT(OnSettingEraserSizeFinished()));
}


//-----------------------------------------------------------------------------
PaintbrushToolGUI::~PaintbrushToolGUI()
{
  if (m_PaintbrushTool.IsNotNull())
  {
    m_PaintbrushTool->EraserSizeChanged -= mitk::MessageDelegate1<PaintbrushToolGUI, int>(this, &PaintbrushToolGUI::OnEraserSizeChangedInTool);
  }
}


//-----------------------------------------------------------------------------
void PaintbrushToolGUI::OnNewToolAssociated(mitk::Tool* tool)
{
  if (m_PaintbrushTool.IsNotNull())
  {
    m_PaintbrushTool->EraserSizeChanged -= mitk::MessageDelegate1<PaintbrushToolGUI, int>(this, &PaintbrushToolGUI::OnEraserSizeChangedInTool);
  }

  m_PaintbrushTool = dynamic_cast<PaintbrushTool*>( tool );

  if (m_PaintbrushTool.IsNotNull())
  {
    m_PaintbrushTool->EraserSizeChanged += mitk::MessageDelegate1<PaintbrushToolGUI, int>(this, &PaintbrushToolGUI::OnEraserSizeChangedInTool);
  }
}


//-----------------------------------------------------------------------------
void PaintbrushToolGUI::OnEraserSizeChangedInGui(double value)
{
  if (m_PaintbrushTool.IsNotNull())
  {
    int eraserSize = static_cast<int>(value + 0.5);

    /// Making sure that the eraser size is odd number so that we can draw a nice cross.
    if (eraserSize % 2 == 0)
    {
      ++eraserSize;
    }

    m_PaintbrushTool->SetEraserSize(eraserSize);

    mitk::BaseRenderer* renderer =
        mitk::GlobalInteraction::GetInstance()->GetFocusManager()->GetFocused();

    mitk::Point2D centreInPx;
    centreInPx[0] = renderer->GetSizeX() / 2;
    centreInPx[1] = renderer->GetSizeY() / 2;
    mitk::Point2D centreInMm;
    renderer->GetDisplayGeometry()->DisplayToWorld(centreInPx, centreInMm);

    m_PaintbrushTool->SetEraserPosition(centreInMm);

    m_PaintbrushTool->SetEraserVisible(true, renderer);
    renderer->RequestUpdate();

    m_ShowEraserTimer->start();
  }
}


//-----------------------------------------------------------------------------
void PaintbrushToolGUI::OnSettingEraserSizeFinished()
{
  mitk::BaseRenderer* renderer =
      mitk::GlobalInteraction::GetInstance()->GetFocusManager()->GetFocused();

  m_PaintbrushTool->SetEraserVisible(false, renderer);

  renderer->RequestUpdate();
}


//-----------------------------------------------------------------------------
void PaintbrushToolGUI::OnEraserSizeChangedInTool(int eraserSize)
{
  m_Slider->setValue(eraserSize);
}

}

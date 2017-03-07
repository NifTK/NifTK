/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkDrawToolGUI.h"

#include <ctkSliderWidget.h>

#include <QLabel>
#include <QLayout>
#include <QPainter>
#include <QTimer>

#include <mitkFocusManager.h>
#include <mitkGlobalInteraction.h>

#include <niftkToolFactoryMacros.h>

namespace niftk
{

NIFTK_TOOL_GUI_MACRO_NO_EXPORT(DrawTool, DrawToolGUI, "Draw Tool GUI")

//-----------------------------------------------------------------------------
DrawToolGUI::DrawToolGUI()
  : QmitkToolGUI(),
    m_Slider(nullptr),
    m_Frame(nullptr),
    m_ShowEraserTimer(new QTimer(this))
{
  // create the visible widgets
  QBoxLayout* layout = new QHBoxLayout(this);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->setSpacing(3);
  this->setLayout(layout);

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

  this->connect(m_Slider, SIGNAL(valueChanged(double)), SLOT(OnEraserSizeChangedInGui(double)));
  this->connect(this, SIGNAL(NewToolAssociated(mitk::Tool*)), SLOT(OnNewToolAssociated(mitk::Tool*)));

  m_ShowEraserTimer->setInterval(600);
  m_ShowEraserTimer->setSingleShot(true);

  this->connect(m_ShowEraserTimer, SIGNAL(timeout()), SLOT(OnSettingEraserSizeFinished()));
}


//-----------------------------------------------------------------------------
DrawToolGUI::~DrawToolGUI()
{
  if (m_DrawTool.IsNotNull())
  {
    m_DrawTool->EraserSizeChanged -= mitk::MessageDelegate1<DrawToolGUI, double>(this, &DrawToolGUI::OnEraserSizeChangedInTool);
  }
}


//-----------------------------------------------------------------------------
void DrawToolGUI::OnNewToolAssociated(mitk::Tool* tool)
{
  if (m_DrawTool.IsNotNull())
  {
    m_DrawTool->EraserSizeChanged -= mitk::MessageDelegate1<DrawToolGUI, double>(this, &DrawToolGUI::OnEraserSizeChangedInTool);
  }

  m_DrawTool = dynamic_cast<DrawTool*>(tool);

  if (m_DrawTool.IsNotNull())
  {
    this->OnEraserSizeChangedInTool(m_DrawTool->GetEraserSize());
    m_DrawTool->EraserSizeChanged += mitk::MessageDelegate1<DrawToolGUI, double>(this, &DrawToolGUI::OnEraserSizeChangedInTool);
  }
}


//-----------------------------------------------------------------------------
void DrawToolGUI::OnEraserSizeChangedInGui(double value)
{
  if (m_DrawTool.IsNotNull())
  {
    m_DrawTool->SetEraserSize(value);

    mitk::BaseRenderer* renderer =
        mitk::GlobalInteraction::GetInstance()->GetFocusManager()->GetFocused();

    mitk::Point2D centreInPx;
    centreInPx[0] = renderer->GetSizeX() / 2;
    centreInPx[1] = renderer->GetSizeY() / 2;
    mitk::Point2D centreInMm;
    renderer->GetDisplayGeometry()->DisplayToWorld(centreInPx, centreInMm);

    m_DrawTool->SetEraserPosition(centreInMm);

    m_DrawTool->SetEraserVisible(true, renderer);
    renderer->RequestUpdate();

    m_ShowEraserTimer->start();
  }
}


//-----------------------------------------------------------------------------
void DrawToolGUI::OnSettingEraserSizeFinished()
{
  mitk::BaseRenderer* renderer =
      mitk::GlobalInteraction::GetInstance()->GetFocusManager()->GetFocused();

  m_DrawTool->SetEraserVisible(false, renderer);

  renderer->RequestUpdate();
}


//-----------------------------------------------------------------------------
void DrawToolGUI::OnEraserSizeChangedInTool(double eraserSize)
{
  m_Slider->setValue(eraserSize);
}

}

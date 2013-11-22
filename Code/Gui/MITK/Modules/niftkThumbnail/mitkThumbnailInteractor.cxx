/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkThumbnailInteractor.h"

#include <string.h>

#include <mitkBaseRenderer.h>
#include <mitkInteractionPositionEvent.h>
#include <mitkLine.h>
#include <mitkSliceNavigationController.h>

#include <QmitkThumbnailRenderWindow.h>

mitk::ThumbnailInteractor::ThumbnailInteractor(QmitkThumbnailRenderWindow* thumbnailWindow)
: mitk::DisplayInteractor()
, m_ThumbnailWindow(thumbnailWindow)
, m_ZoomFactor(1.05)
{
  m_Renderer = thumbnailWindow->GetRenderer();
  m_SliceNavigationController = m_Renderer->GetSliceNavigationController();
  m_StartDisplayCoordinate.Fill(0);
  m_StartCoordinateInMM.Fill(0);
  m_LastDisplayCoordinate.Fill(0);
  m_CurrentDisplayCoordinate.Fill(0);
}

mitk::ThumbnailInteractor::~ThumbnailInteractor()
{
}

void mitk::ThumbnailInteractor::Notify(InteractionEvent* interactionEvent, bool isHandled)
{
  mitk::BaseRenderer* renderer = interactionEvent->GetSender();
  if (m_Renderer == renderer)
  {
    Superclass::Notify(interactionEvent, isHandled);
  }
}

void mitk::ThumbnailInteractor::ConnectActionsAndFunctions()
{
//  mitk::DisplayInteractor::ConnectActionsAndFunctions();
  CONNECT_FUNCTION("init", Init);
  CONNECT_FUNCTION("initZoom", Init);
  CONNECT_FUNCTION("move", Move);
  CONNECT_FUNCTION("zoom", Zoom);
}

bool mitk::ThumbnailInteractor::Init(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  BaseRenderer* renderer = interactionEvent->GetSender();
  InteractionPositionEvent* positionEvent = static_cast<InteractionPositionEvent*>(interactionEvent);

  Vector2D origin = renderer->GetDisplayGeometry()->GetOriginInMM();
  double scaleFactorMMPerDisplayUnit = renderer->GetDisplayGeometry()->GetScaleFactorMMPerDisplayUnit();
  m_StartDisplayCoordinate = positionEvent->GetPointerPositionOnScreen();
  m_LastDisplayCoordinate = positionEvent->GetPointerPositionOnScreen();
  m_CurrentDisplayCoordinate = positionEvent->GetPointerPositionOnScreen();
  m_StartCoordinateInMM = mitk::Point2D(
      (origin + m_StartDisplayCoordinate.GetVectorFromOrigin() * scaleFactorMMPerDisplayUnit).GetDataPointer());

  return true;
}

bool mitk::ThumbnailInteractor::InitZoom(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  mitk::InteractionPositionEvent* positionEvent = static_cast<mitk::InteractionPositionEvent*>(interactionEvent);

  m_ThumbnailWindow->OnSelectedPositionChanged(positionEvent->GetPositionInWorld());

  return this->Init(action, interactionEvent);
}

bool mitk::ThumbnailInteractor::Move(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  InteractionPositionEvent* positionEvent = static_cast<InteractionPositionEvent*>(interactionEvent);

  mitk::Point2D displayCoordinate = positionEvent->GetPointerPositionOnScreen();
  mitk::Vector2D displacement = displayCoordinate - m_LastDisplayCoordinate;
  m_LastDisplayCoordinate = displayCoordinate;

  m_ThumbnailWindow->OnBoundingBoxPanned(displacement);

  return true;
}

bool mitk::ThumbnailInteractor::Zoom(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  double scaleFactor = 1.0;

  InteractionPositionEvent* positionEvent = static_cast<InteractionPositionEvent*>(interactionEvent);

  float distance = m_CurrentDisplayCoordinate[1] - m_LastDisplayCoordinate[1];

  // set zooming speed
  if (distance < 0.0)
  {
    scaleFactor = 1.0 / m_ZoomFactor;
  }
  else if (distance > 0.0)
  {
    scaleFactor = 1.0 * m_ZoomFactor;
  }

  m_LastDisplayCoordinate = m_CurrentDisplayCoordinate;
  m_CurrentDisplayCoordinate = positionEvent->GetPointerPositionOnScreen();

  m_ThumbnailWindow->OnBoundingBoxZoomed(scaleFactor, m_StartCoordinateInMM);

  return true;
}

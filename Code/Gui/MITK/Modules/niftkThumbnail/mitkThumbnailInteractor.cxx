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

mitk::ThumbnailInteractor::ThumbnailInteractor(mitk::BaseRenderer* renderer)
: mitk::DisplayInteractor()
, m_Renderer(renderer)
{
  m_SliceNavigationController = renderer->GetSliceNavigationController();
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
  CONNECT_FUNCTION("initZoom", InitZoom);
  CONNECT_FUNCTION("move", Move);
  CONNECT_FUNCTION("zoom", Zoom);
}

bool mitk::ThumbnailInteractor::Init(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  return true;
}

bool mitk::ThumbnailInteractor::InitZoom(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  MITK_INFO << "mitk::ThumbnailInteractor::InitZoom(StateMachineAction* action, InteractionEvent* interactionEvent)";
  return true;
//  BaseRenderer* renderer = interactionEvent->GetSender();
//  InteractionPositionEvent* positionEvent = dynamic_cast<InteractionPositionEvent*>(interactionEvent);
//  if (positionEvent == NULL)
//  {
//    MITK_WARN << "mitk ThumbnailInteractor cannot process the event: " << interactionEvent->GetNameOfClass();
//    return false;
//  }

//  // First, check if the slice navigation controllers have a valid geometry,
//  // i.e. an image is loaded.
//  if (!m_SliceNavigationController->GetCreatedWorldGeometry())
//  {
//    return false;
//  }

//  return this->Init(action, positionEvent2);
}

bool mitk::ThumbnailInteractor::Move(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  MITK_INFO << "mitk::ThumbnailInteractor::Move(StateMachineAction* action, InteractionEvent* interactionEvent)";
}

bool mitk::ThumbnailInteractor::Zoom(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  MITK_INFO << "mitk::ThumbnailInteractor::Zoom(StateMachineAction* action, InteractionEvent* interactionEvent)";
}

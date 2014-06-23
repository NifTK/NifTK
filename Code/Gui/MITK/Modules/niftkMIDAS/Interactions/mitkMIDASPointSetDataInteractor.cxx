/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASPointSetDataInteractor.h"

#include <mitkPositionEvent.h>
#include <mitkBaseRenderer.h>
#include <mitkRenderingManager.h>
#include <mitkPointSet.h>
#include <mitkInteractionConst.h>
#include <mitkInteractionPositionEvent.h>

mitk::MIDASPointSetDataInteractor::MIDASPointSetDataInteractor()
: mitk::PointSetDataInteractor()
{
  this->SetAccuracy(1.0);
}

mitk::MIDASPointSetDataInteractor::~MIDASPointSetDataInteractor()
{
}

bool mitk::MIDASPointSetDataInteractor::FilterEvents(mitk::InteractionEvent* event, mitk::DataNode* dataNode)
{
  return MIDASStateMachine::CanHandleEvent(event);
}

//##Documentation
//## overwritten cause this class can handle it better!
bool mitk::MIDASPointSetDataInteractor::CanHandle(mitk::InteractionEvent* event)
{
/*
  float returnValue = 0.0f;

  //if it is a key event that can be handled in the current state, then return 0.5
  mitk::InteractionPositionEvent* displayPositionEvent =
    dynamic_cast<mitk::InteractionPositionEvent*>(event);

  // Key event handling:
  if (!displayPositionEvent)
  {
    // Check, if the current state has a transition waiting for that key event.
    if (this->GetCurrentState()->GetTransition(stateEvent->GetId()))
    {
      return 0.5f;
    }
    else
    {
      return 0.0f;
    }
  }

  // Get the time of the sender to look for the right transition.
  mitk::BaseRenderer* renderer = stateEvent->GetEvent()->GetSender();
  if (renderer)
  {
    unsigned int timeStep = renderer->GetTimeStep(m_DataNode->GetData());

    // If the event can be understood and if there is a transition waiting for that event
    mitk::State const* state = this->GetCurrentState(timeStep);
    if (state)
    {
      if (state->GetTransition(stateEvent->GetId()))
      {
        returnValue = 0.5; //it can be understood
      }
    }

    mitk::PointSet* pointSet = dynamic_cast<mitk::PointSet*>(m_DataNode->GetData());
    if (pointSet)
    {
      // if we have one point or more, then check if the have been picked
      if (pointSet->GetSize(timeStep) > 0
          && pointSet->SearchPoint(displayPositionEvent->GetWorldPosition(), m_Precision, timeStep) > -1)
      {
        returnValue = 1.0;
      }
    }
  }
  return returnValue;
*/

  return true;
}

bool mitk::MIDASPointSetDataInteractor::ExecuteAction(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  mitk::InteractionPositionEvent* positionEvent =
      dynamic_cast<mitk::InteractionPositionEvent*>(event);

  if (positionEvent)
  {
    mitk::BaseRenderer* renderer = positionEvent->GetSender();

    mitk::Point3D point3DInMm = positionEvent->GetPositionInWorld();
    const mitk::Geometry3D* worldGeometry = renderer->GetWorldGeometry();
    mitk::Point3D point3DIndex;
    worldGeometry->WorldToIndex(point3DInMm, point3DIndex);
    point3DIndex[0] = std::floor(point3DIndex[0]) + 0.5;
    point3DIndex[1] = std::floor(point3DIndex[1]) + 0.5;
    worldGeometry->IndexToWorld(point3DIndex, point3DInMm);

    mitk::Point2D point2DInMm;
    mitk::Point2D point2DInPx;

    mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();
    displayGeometry->Map(point3DInMm, point2DInMm);
    displayGeometry->WorldToDisplay(point2DInMm, point2DInPx);

    /// TODO Disabled during the MITK upgrade.
//    positionEvent->SetDisplayPosition(point2DInPx);
  }

  return Superclass::ExecuteAction(action, event);
}

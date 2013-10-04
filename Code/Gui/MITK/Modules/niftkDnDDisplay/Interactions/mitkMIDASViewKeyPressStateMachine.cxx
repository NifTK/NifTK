/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASViewKeyPressStateMachine.h"
#include <mitkWheelEvent.h>
#include <mitkStateEvent.h>
#include <mitkBaseRenderer.h>

namespace mitk
{

const std::string mitk::MIDASViewKeyPressStateMachine::STATE_MACHINE_XML =
    "<stateMachine NAME=\"MIDASViewKeyPressStateMachine\">"
    "  <state NAME=\"stateStart\"  START_STATE=\"TRUE\"   ID=\"1\" X_POS=\"50\"   Y_POS=\"100\" WIDTH=\"100\" HEIGHT=\"50\">"
    "    <transition NAME=\"keyPressA\" NEXT_STATE_ID=\"1\" EVENT_ID=\"4001\">"
    "      <action ID=\"350001\" />"
    "    </transition>"
    "    <transition NAME=\"keyPressZ\" NEXT_STATE_ID=\"1\" EVENT_ID=\"4019\">"
    "      <action ID=\"350002\" />"
    "    </transition>"
    "    <transition NAME=\"keyPressQ\" NEXT_STATE_ID=\"1\" EVENT_ID=\"4013\">"
    "      <action ID=\"350003\" />"
    "    </transition>"
    "    <transition NAME=\"keyPressW\" NEXT_STATE_ID=\"1\" EVENT_ID=\"4016\">"
    "      <action ID=\"350004\" />"
    "    </transition>"
    "    <transition NAME=\"keyPressE\" NEXT_STATE_ID=\"1\" EVENT_ID=\"19\">"
    "      <action ID=\"350005\" />"
    "    </transition>"
    "    <transition NAME=\"mouseButtonLeftDoubleClick\" NEXT_STATE_ID=\"1\" EVENT_ID=\"8\">"
    "      <action ID=\"350013\" />"
    "    </transition>"
    "  </state>"
    "</stateMachine>";

//-----------------------------------------------------------------------------
MIDASViewKeyPressStateMachine::MIDASViewKeyPressStateMachine(const char* stateMachinePattern, MIDASViewKeyPressResponder* responder)
: StateMachine(stateMachinePattern)
{
  assert(responder);
  m_Responder = responder;

  CONNECT_ACTION(350001, MoveAnterior);
  CONNECT_ACTION(350002, MovePosterior);
  CONNECT_ACTION(350003, SwitchToAxial);
  CONNECT_ACTION(350004, SwitchToSagittal);
  CONNECT_ACTION(350005, SwitchToCoronal);
  CONNECT_ACTION(350013, ToggleMultiWindowLayout);
}


//-----------------------------------------------------------------------------
bool MIDASViewKeyPressStateMachine::HandleEvent(StateEvent const* stateEvent)
{
  mitk::BaseRenderer* sender = stateEvent->GetEvent()->GetSender();
  if (!this->HasRenderer(sender))
  {
    return false;
  }
  return mitk::StateMachine::HandleEvent(stateEvent);
}


//-----------------------------------------------------------------------------
bool MIDASViewKeyPressStateMachine::HasRenderer(const mitk::BaseRenderer* renderer) const
{
  std::vector<const mitk::BaseRenderer*>::const_iterator begin = m_Renderers.begin();
  std::vector<const mitk::BaseRenderer*>::const_iterator end = m_Renderers.end();

  return std::find(begin, end, renderer) != end;
}


//-----------------------------------------------------------------------------
void MIDASViewKeyPressStateMachine::AddRenderer(const mitk::BaseRenderer* renderer)
{
  if (!this->HasRenderer(renderer))
  {
    m_Renderers.push_back(renderer);
  }
}


//-----------------------------------------------------------------------------
void MIDASViewKeyPressStateMachine::RemoveRenderer(const mitk::BaseRenderer* renderer)
{
  std::vector<const mitk::BaseRenderer*>::iterator begin = m_Renderers.begin();
  std::vector<const mitk::BaseRenderer*>::iterator end = m_Renderers.end();

  std::vector<const mitk::BaseRenderer*>::iterator foundRenderer = std::find(begin, end, renderer);
  if (foundRenderer != end)
  {
    m_Renderers.erase(foundRenderer);
  }
}


//-----------------------------------------------------------------------------
float MIDASViewKeyPressStateMachine::CanHandleEvent(const StateEvent *event) const
{
  // See StateMachine.xml for event Ids.
  if (event != NULL
      && event->GetEvent() != NULL
      && (   event->GetId() == 4001 // A
          || event->GetId() == 4019 // Z
          || event->GetId() == 4013 // Q
          || event->GetId() == 4016 // W
          || event->GetId() == 19   // E
          || event->GetId() == 8    // left mouse button double click
          )
      )
  {
    return 1.0f;
  }

  return mitk::StateMachine::CanHandleEvent(event);
}


//-----------------------------------------------------------------------------
bool MIDASViewKeyPressStateMachine::MoveAnterior(Action*, const StateEvent*)
{
  return m_Responder->MoveAnterior();
}


//-----------------------------------------------------------------------------
bool MIDASViewKeyPressStateMachine::MovePosterior(Action*, const StateEvent*)
{
  return m_Responder->MovePosterior();
}


//-----------------------------------------------------------------------------
bool MIDASViewKeyPressStateMachine::SwitchToAxial(Action*, const StateEvent*)
{
  return m_Responder->SwitchToAxial();
}


//-----------------------------------------------------------------------------
bool MIDASViewKeyPressStateMachine::SwitchToSagittal(Action*, const StateEvent*)
{
  return m_Responder->SwitchToSagittal();
}


//-----------------------------------------------------------------------------
bool MIDASViewKeyPressStateMachine::SwitchToCoronal(Action*, const StateEvent*)
{
  return m_Responder->SwitchToCoronal();
}

//-----------------------------------------------------------------------------
bool MIDASViewKeyPressStateMachine::ToggleMultiWindowLayout(Action*, const StateEvent*)
{
  return m_Responder->ToggleMultiWindowLayout();
}

} // end namespace

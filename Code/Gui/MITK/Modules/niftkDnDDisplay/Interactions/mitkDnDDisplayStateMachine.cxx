/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkDnDDisplayStateMachine.h"
#include <mitkWheelEvent.h>
#include <mitkStateEvent.h>
#include <mitkBaseRenderer.h>
#include <mitkGlobalInteraction.h>

namespace mitk
{

//-----------------------------------------------------------------------------
const std::string mitk::DnDDisplayStateMachine::STATE_MACHINE_XML =
    "<stateMachine NAME=\"DnDDisplayStateMachine\">"
    "  <state NAME=\"stateStart\" ID=\"1\" START_STATE=\"TRUE\">"
    "    <transition NAME=\"keyPressK\" EVENT_ID=\"4009\" NEXT_STATE_ID=\"1\">"
    "      <action ID=\"350015\"/>"
    "    </transition>"
    "    <transition NAME=\"keyPressL\" EVENT_ID=\"4010\" NEXT_STATE_ID=\"1\">"
    "      <action ID=\"350016\"/>"
    "    </transition>"
    "  </state>"
    "</stateMachine>";


//-----------------------------------------------------------------------------
bool DnDDisplayStateMachine::s_BehaviourStringLoaded = false;


//-----------------------------------------------------------------------------
DnDDisplayStateMachine::DnDDisplayStateMachine(const char* stateMachinePattern, DnDDisplayStateMachineResponder* responder)
: StateMachine(stateMachinePattern)
{
  assert(responder);

  m_Responder = responder;
}


//-----------------------------------------------------------------------------
void DnDDisplayStateMachine::LoadBehaviourString()
{
  if (!s_BehaviourStringLoaded)
  {
    mitk::GlobalInteraction* globalInteraction =  mitk::GlobalInteraction::GetInstance();
    mitk::StateMachineFactory* stateMachineFactory = globalInteraction->GetStateMachineFactory();
    if (stateMachineFactory)
    {
      if (stateMachineFactory->LoadBehaviorString(mitk::DnDDisplayStateMachine::STATE_MACHINE_XML))
      {
        s_BehaviourStringLoaded = true;
      }
    }
    else
    {
      MITK_ERROR << "State machine factory is not initialised. Use QmitkRegisterClasses().";
    }
  }
}


//-----------------------------------------------------------------------------
bool DnDDisplayStateMachine::HandleEvent(StateEvent const* stateEvent)
{
  mitk::BaseRenderer* sender = stateEvent->GetEvent()->GetSender();
  if (!this->HasRenderer(sender))
  {
    return false;
  }
  return mitk::StateMachine::HandleEvent(stateEvent);
}


//-----------------------------------------------------------------------------
bool DnDDisplayStateMachine::HasRenderer(const mitk::BaseRenderer* renderer) const
{
  std::vector<const mitk::BaseRenderer*>::const_iterator begin = m_Renderers.begin();
  std::vector<const mitk::BaseRenderer*>::const_iterator end = m_Renderers.end();

  return std::find(begin, end, renderer) != end;
}


//-----------------------------------------------------------------------------
void DnDDisplayStateMachine::AddRenderer(const mitk::BaseRenderer* renderer)
{
  if (!this->HasRenderer(renderer))
  {
    m_Renderers.push_back(renderer);
  }
}


//-----------------------------------------------------------------------------
void DnDDisplayStateMachine::RemoveRenderer(const mitk::BaseRenderer* renderer)
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
float DnDDisplayStateMachine::CanHandleEvent(const StateEvent *event) const
{
  // See StateMachine.xml for event Ids.
  if (event != NULL
      && event->GetEvent() != NULL
      )
  {
    return 1.0f;
  }

  return mitk::StateMachine::CanHandleEvent(event);
}


//-----------------------------------------------------------------------------
bool DnDDisplayStateMachine::MoveAnterior(Action*, const StateEvent*)
{
  return m_Responder->MoveAnterior();
}


//-----------------------------------------------------------------------------
bool DnDDisplayStateMachine::MovePosterior(Action*, const StateEvent*)
{
  return m_Responder->MovePosterior();
}


//-----------------------------------------------------------------------------
bool DnDDisplayStateMachine::SelectPreviousTimeStep(Action*, const StateEvent*)
{
  return m_Responder->SelectPreviousTimeStep();
}


//-----------------------------------------------------------------------------
bool DnDDisplayStateMachine::SelectNextTimeStep(Action*, const StateEvent*)
{
  return m_Responder->SelectNextTimeStep();
}


//-----------------------------------------------------------------------------
bool DnDDisplayStateMachine::SwitchToAxial(Action*, const StateEvent*)
{
  return m_Responder->SwitchToAxial();
}


//-----------------------------------------------------------------------------
bool DnDDisplayStateMachine::SwitchToSagittal(Action*, const StateEvent*)
{
  return m_Responder->SwitchToSagittal();
}


//-----------------------------------------------------------------------------
bool DnDDisplayStateMachine::SwitchToCoronal(Action*, const StateEvent*)
{
  return m_Responder->SwitchToCoronal();
}

//-----------------------------------------------------------------------------
bool DnDDisplayStateMachine::ToggleMultiWindowLayout(Action*, const StateEvent*)
{
  return m_Responder->ToggleMultiWindowLayout();
}


//-----------------------------------------------------------------------------
bool DnDDisplayStateMachine::ToggleCursorVisibility(Action*, const StateEvent*)
{
  return m_Responder->ToggleCursorVisibility();
}

}

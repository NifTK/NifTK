/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASToolKeyPressStateMachine.h"
#include <mitkWheelEvent.h>

#include <usGetModuleContext.h>


namespace mitk
{

//-----------------------------------------------------------------------------
void MIDASToolKeyPressStateMachine::Notify(mitk::InteractionEvent* interactionEvent, bool isHandled)
{
  // to use the state machine pattern,
  // the event is passed to the state machine interface to be handled
  if (!isHandled)
  {
    this->HandleEvent(interactionEvent, NULL);
  }
}


//-----------------------------------------------------------------------------
MIDASToolKeyPressStateMachine::MIDASToolKeyPressStateMachine(MIDASToolKeyPressResponder* responder)
: mitk::EventStateMachine()
{
  assert(responder);
  m_Responder = responder;

  this->LoadStateMachine("MIDASToolKeyPressStateMachine.xml", us::GetModuleContext()->GetModule());
  this->SetEventConfig("MIDASToolKeyPressStateMachineConfig.xml", us::GetModuleContext()->GetModule());

  // Register as listener via micro services
  us::ServiceProperties props;
  props["name"] = std::string("MIDASToolKeyPressStateMachine");

  m_ServiceRegistration = us::GetModuleContext()->RegisterService<mitk::InteractionEventObserver>(this, props);
}


//-----------------------------------------------------------------------------
MIDASToolKeyPressStateMachine::~MIDASToolKeyPressStateMachine()
{
  m_ServiceRegistration.Unregister();
}


//-----------------------------------------------------------------------------
void mitk::MIDASToolKeyPressStateMachine::ConnectActionsAndFunctions()
{
  CONNECT_FUNCTION("selectSeedTool", SelectSeedTool);
  CONNECT_FUNCTION("selectDrawTool", SelectDrawTool);
  CONNECT_FUNCTION("unselectTools", UnselectTools);
  CONNECT_FUNCTION("selectPolyTool", SelectPolyTool);
  CONNECT_FUNCTION("selectViewMode", SelectViewMode);
  CONNECT_FUNCTION("cleanSlice", CleanSlice);
}


//-----------------------------------------------------------------------------
bool mitk::MIDASToolKeyPressStateMachine::FilterEvents(mitk::InteractionEvent* event, mitk::DataNode* dataNode)
{
  return this->CanHandleEvent(event);
}


//-----------------------------------------------------------------------------
bool MIDASToolKeyPressStateMachine::SelectSeedTool(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  return m_Responder->SelectSeedTool();
}


//-----------------------------------------------------------------------------
bool MIDASToolKeyPressStateMachine::SelectDrawTool(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  return m_Responder->SelectDrawTool();
}


//-----------------------------------------------------------------------------
bool MIDASToolKeyPressStateMachine::UnselectTools(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  return m_Responder->UnselectTools();
}


//-----------------------------------------------------------------------------
bool MIDASToolKeyPressStateMachine::SelectPolyTool(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  return m_Responder->SelectPolyTool();
}


//-----------------------------------------------------------------------------
bool MIDASToolKeyPressStateMachine::SelectViewMode(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  return m_Responder->SelectViewMode();
}


//-----------------------------------------------------------------------------
bool MIDASToolKeyPressStateMachine::CleanSlice(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  return m_Responder->CleanSlice();
}

}

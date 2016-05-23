/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkToolKeyPressStateMachine.h"

#include <mitkWheelEvent.h>

#include <usGetModuleContext.h>


namespace niftk
{

//-----------------------------------------------------------------------------
void ToolKeyPressStateMachine::Notify(mitk::InteractionEvent* interactionEvent, bool isHandled)
{
  // to use the state machine pattern,
  // the event is passed to the state machine interface to be handled
  if (!isHandled)
  {
    this->HandleEvent(interactionEvent, NULL);
  }
}


//-----------------------------------------------------------------------------
ToolKeyPressStateMachine::ToolKeyPressStateMachine(ToolKeyPressResponder* responder)
: mitk::EventStateMachine()
{
  assert(responder);
  m_Responder = responder;

  this->LoadStateMachine("niftkToolKeyPressStateMachine.xml", us::GetModuleContext()->GetModule());
  this->SetEventConfig("niftkToolKeyPressStateMachineConfig.xml", us::GetModuleContext()->GetModule());

  // Register as listener via micro services
  us::ServiceProperties props;
  props["name"] = std::string("niftkToolKeyPressStateMachine");

  m_ServiceRegistration = us::GetModuleContext()->RegisterService<mitk::InteractionEventObserver>(this, props);
}


//-----------------------------------------------------------------------------
ToolKeyPressStateMachine::~ToolKeyPressStateMachine()
{
  m_ServiceRegistration.Unregister();
}


//-----------------------------------------------------------------------------
void ToolKeyPressStateMachine::ConnectActionsAndFunctions()
{
  CONNECT_FUNCTION("selectSeedTool", SelectSeedTool);
  CONNECT_FUNCTION("selectDrawTool", SelectDrawTool);
  CONNECT_FUNCTION("unselectTools", UnselectTools);
  CONNECT_FUNCTION("selectPolyTool", SelectPolyTool);
  CONNECT_FUNCTION("selectViewMode", SelectViewMode);
  CONNECT_FUNCTION("cleanSlice", CleanSlice);
}


//-----------------------------------------------------------------------------
bool ToolKeyPressStateMachine::FilterEvents(mitk::InteractionEvent* event, mitk::DataNode* dataNode)
{
  return this->CanHandleEvent(event);
}


//-----------------------------------------------------------------------------
bool ToolKeyPressStateMachine::SelectSeedTool(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  return m_Responder->SelectSeedTool();
}


//-----------------------------------------------------------------------------
bool ToolKeyPressStateMachine::SelectDrawTool(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  return m_Responder->SelectDrawTool();
}


//-----------------------------------------------------------------------------
bool ToolKeyPressStateMachine::UnselectTools(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  return m_Responder->UnselectTools();
}


//-----------------------------------------------------------------------------
bool ToolKeyPressStateMachine::SelectPolyTool(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  return m_Responder->SelectPolyTool();
}


//-----------------------------------------------------------------------------
bool ToolKeyPressStateMachine::SelectViewMode(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  return m_Responder->SelectViewMode();
}


//-----------------------------------------------------------------------------
bool ToolKeyPressStateMachine::CleanSlice(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  return m_Responder->CleanSlice();
}

}

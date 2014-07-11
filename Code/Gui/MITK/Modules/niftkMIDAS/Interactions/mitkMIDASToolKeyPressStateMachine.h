/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkMIDASToolKeyPressStateMachine_h
#define mitkMIDASToolKeyPressStateMachine_h

#include "niftkMIDASExports.h"
#include "mitkMIDASToolKeyPressResponder.h"
#include <mitkStateMachine.h>

#include "mitkMIDASStateMachine.h"

namespace mitk {

/**
 * \class MIDASToolPressStateMachine
 * \brief StateMachine to check for key press events that MIDAS is interested in,
 * and pass them onto any registered MIDASToolKeyPressResponder.
 */
class NIFTKMIDAS_EXPORT MIDASToolKeyPressStateMachine : public mitk::EventStateMachine, public MIDASStateMachine
{

public:
  mitkClassMacro(MIDASToolKeyPressStateMachine, mitk::EventStateMachine); // this creates the Self typedef
  mitkNewMacro1Param(Self, MIDASToolKeyPressResponder*);

protected:

  /// \brief Purposely hidden, protected constructor, so class is instantiated via static ::New() macro, where we pass in the state machine pattern name, and also the object to pass the data onto.
  MIDASToolKeyPressStateMachine(MIDASToolKeyPressResponder* responder);

  /// \brief Purposely hidden, destructor.
  virtual ~MIDASToolKeyPressStateMachine();

  /// \brief Connects state machine actions to functions.
  virtual void ConnectActionsAndFunctions();

  /// \brief Tells if this tool can handle the given event.
  ///
  /// This implementation delegates the call to mitk::MIDASStateMachine::CanHandleEvent(),
  /// that checks if the event is filtered by one of the installed event filters and if not,
  /// calls CanHandle() and returns with its result.
  ///
  /// Note that this function is purposefully not virtual. Eventual subclasses should
  /// override the CanHandle function.
  virtual bool FilterEvents(mitk::InteractionEvent* event, mitk::DataNode* dataNode);

  /// \see mitk::MIDASToolKeyPressResponder::SelectSeedTool()
  bool SelectSeedTool(mitk::StateMachineAction* action, mitk::InteractionEvent* event);

  /// \see mitk::MIDASToolKeyPressResponder::SelectDrawTool()
  bool SelectDrawTool(mitk::StateMachineAction* action, mitk::InteractionEvent* event);

  /// \see mitk::MIDASToolKeyPressResponder::UnselectTools()
  bool UnselectTools(mitk::StateMachineAction* action, mitk::InteractionEvent* event);

  /// \see mitk::MIDASToolKeyPressResponder::SelectPolyTool()
  bool SelectPolyTool(mitk::StateMachineAction* action, mitk::InteractionEvent* event);

  /// \see mitk::MIDASToolKeyPressResponder::SelectViewMode()
  bool SelectViewMode(mitk::StateMachineAction* action, mitk::InteractionEvent* event);

  /// \see mitk::MIDASToolKeyPressResponder::CleanSlice()
  bool CleanSlice(mitk::StateMachineAction* action, mitk::InteractionEvent* event);

private:

  /// \brief the object that gets called, specified in constructor.
  MIDASToolKeyPressResponder* m_Responder;

};

}

#endif

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkToolKeyPressStateMachine_h
#define niftkToolKeyPressStateMachine_h

#include "niftkMIDASExports.h"

#include <mitkInteractionEventObserver.h>
#include <mitkStateMachine.h>

#include <usServiceRegistration.h>

#include "niftkFilteringStateMachine.h"
#include "niftkToolKeyPressResponder.h"

namespace niftk
{

/**
 * \class ToolPressStateMachine
 * \brief StateMachine to check for key press events that MIDAS is interested in,
 * and pass them onto any registered ToolKeyPressResponder.
 */
class NIFTKMIDAS_EXPORT ToolKeyPressStateMachine : public mitk::EventStateMachine, public mitk::InteractionEventObserver, public FilteringStateMachine
{

public:
  mitkClassMacro(ToolKeyPressStateMachine, mitk::EventStateMachine) // this creates the Self typedef
  mitkNewMacro1Param(Self, ToolKeyPressResponder*)

  /**
   * By this function the Observer gets notified about new events.
   * Here it is adapted to pass the events to the state machine in order to use
   * its infrastructure.
   * It also checks if event is to be accepted when it already has been processed by a DataInteractor.
   */
  virtual void Notify(mitk::InteractionEvent* interactionEvent, bool isHandled) override;

protected:

  /// \brief Purposely hidden, protected constructor, so class is instantiated via static ::New() macro, where we pass in the state machine pattern name, and also the object to pass the data onto.
  ToolKeyPressStateMachine(ToolKeyPressResponder* responder);

  /// \brief Purposely hidden, destructor.
  virtual ~ToolKeyPressStateMachine();

  /// \brief Connects state machine actions to functions.
  virtual void ConnectActionsAndFunctions() override;

  /// \brief Tells if this tool can handle the given event.
  ///
  /// This implementation delegates the call to FilteringStateMachine::CanHandleEvent(),
  /// that checks if the event is filtered by one of the installed event filters and if not,
  /// calls CanHandle() and returns with its result.
  ///
  /// Note that this function is purposefully not virtual. Eventual subclasses should
  /// override the CanHandle function.
  virtual bool FilterEvents(mitk::InteractionEvent* event, mitk::DataNode* dataNode) override;

  /// \see ToolKeyPressResponder::SelectSeedTool()
  bool SelectSeedTool(mitk::StateMachineAction* action, mitk::InteractionEvent* event);

  /// \see ToolKeyPressResponder::SelectDrawTool()
  bool SelectDrawTool(mitk::StateMachineAction* action, mitk::InteractionEvent* event);

  /// \see ToolKeyPressResponder::UnselectTools()
  bool UnselectTools(mitk::StateMachineAction* action, mitk::InteractionEvent* event);

  /// \see ToolKeyPressResponder::SelectPolyTool()
  bool SelectPolyTool(mitk::StateMachineAction* action, mitk::InteractionEvent* event);

  /// \see ToolKeyPressResponder::SelectViewMode()
  bool SelectViewMode(mitk::StateMachineAction* action, mitk::InteractionEvent* event);

  /// \see ToolKeyPressResponder::CleanSlice()
  bool CleanSlice(mitk::StateMachineAction* action, mitk::InteractionEvent* event);

private:

  /// \brief the object that gets called, specified in constructor.
  ToolKeyPressResponder* m_Responder;

  /**
    * Reference to the service registration of the observer,
    * it is needed to unregister the observer on unload.
    */
   us::ServiceRegistration<mitk::InteractionEventObserver> m_ServiceRegistration;

};

}

#endif

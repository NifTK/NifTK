/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkFilteringStateMachine_h
#define niftkFilteringStateMachine_h

#include "niftkMIDASExports.h"

#include <vector>

#include <mitkInteractionEvent.h>
#include <mitkStateEvent.h>

namespace niftk
{

class StateMachineEventFilter;

/**
 * \class FilteringStateMachine
 *
 * \brief Common base class for segmentor tools and interactors.
 *
 * Provides a way to define event filters externally and apply
 * them to state machines. This can be used to discard
 * events that come from an unwanted render window. With other
 * words, we can limit the scope of the state machine to certain
 * render windows, e.g. to the active render window of the main display.
 *
 * The class provides an implementation for the mitk::StateMachine::CanHandleEvent(const mitk::StateEvent* stateEvent)
 * function that first checks if the event is filtered, and if not it calls the
 * protected CanHandle function. Derived classes should override mitk::StateMachine::CanHandleEvent()
 * and delegate the call to the CanHandleEvent() function of this class. They should allow new
 * types of events by overriding the CanHandle function.
 *
 * Note that this class is not derived from mitk::StateMachine.
 */
class NIFTKMIDAS_EXPORT FilteringStateMachine
{

public:

  /// \brief Constructs a FilteringStateMachine object.
  FilteringStateMachine();

  /// \brief Destructs the FilteringStateMachine object.
  virtual ~FilteringStateMachine();

  /// \brief This function is to replace the original CanHandleEvent function to support event filtering.
  ///
  /// Checks if the event is filtered by one of the registered event filters. If yes, it returns 0.
  /// Otherwise, it calls CanHandle(const mitk::StateEvent*) and returns with its result.
  ///
  /// Note that this function is not virtual. Derived classes should override the
  /// mitk::StateMachine::CanHandleEvent() function and delegate the call to this function.
  ///
  /// float CanHandleEvent(const mitk::StateEvent* stateEvent) const
  /// {
  ///   return FilteringStateMachine::CanHandleEvent(stateEvent);
  /// }
  ///
  /// The original logic should be implemented in the CanHandle function.
  ///
  /// \see mitk::StateMachine::CanHandleEvent
  float CanHandleEvent(const mitk::StateEvent* event) const;

  bool CanHandleEvent(mitk::InteractionEvent* event);

  /// \brief Installs an event filter that can reject a state machine event or let it pass through.
  virtual void InstallEventFilter(StateMachineEventFilter* eventFilter);

  /// \brief Removes an event filter that can reject a state machine event or let it pass through.
  virtual void RemoveEventFilter(StateMachineEventFilter* eventFilter);

  /// \brief Gets the list of the installed event filters.
  std::vector<StateMachineEventFilter*> GetEventFilters() const;

  /// \brief Tells if the event is rejected by the installed event filters or they let it pass through.
  bool IsFiltered(const mitk::StateEvent* stateEvent) const;

  /// \brief Tells if the event is rejected by the installed event filters or they let it pass through.
  bool IsFiltered(mitk::InteractionEvent* event);

protected:

  /// Tells if this state machine can handle the event. This function is called only if the event
  /// has not been rejected by one of the event filters.
  /// Derived classes should override this function in favor of mitk::StateMachine::CanHandleEvent
  /// if they can process also other kinds of events than what their base class can do.
  virtual float CanHandle(const mitk::StateEvent* stateEvent) const
  {
    return 1.0;
  }

  virtual bool CanHandle(mitk::InteractionEvent* event)
  {
    return true;
  }

private:

  /// \brief Filter the events that are sent to the interactors.
  std::vector<StateMachineEventFilter*> m_EventFilters;

};

}

#endif

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkMIDASStateMachine_h
#define mitkMIDASStateMachine_h

#include "niftkMIDASExports.h"

#include <mitkStateEvent.h>

#include "mitkMIDASEventFilter.h"

#include <vector>

namespace mitk
{

/**
 * \class MIDASStateMachine
 *
 * \brief Common base class for MIDAS tools and interactors.
 *
 * Provides a way to define event filters externally and apply
 * them to the MIDAS state machines. This can be used to discard
 * events that come from an unwanted render window. With other
 * words, we can limit the scope of the state machine to certain
 * render windows, e.g. our DnD Display.
 *
 * The class provides an implementation of the mitk::StateMachine::CanHandleEvent(const mitk::StateEvent* stateEvent)
 * function that first checks if the event is filtered, and if not it calls the
 * protected CanHandle function. Classes derived from this should allow new
 * types of events by overriding the CanHandle function.
 *
 * Note that this class is not derived from
 */
class NIFTKMIDAS_EXPORT MIDASStateMachine
{

public:

  /// \brief Constructs a MIDASStateMachine object.
  MIDASStateMachine();

  /// \brief Destructs the MIDASStateMachine object.
  virtual ~MIDASStateMachine();

  /// \brief Overrides mitk::StateMachine::CanHandleEvent and enables event filtering.
  ///
  /// Checks if the event is filtered by one of the registered event filters. If yes, it returns 0.
  /// Otherwise, it calls CanHandle(const mitk::StateEvent*) and returns with its result.
  ///
  /// Note that this function is not virtual. Derived classes should override the CanHandle function.
  /// \see mitk::StateMachine::CanHandleEvent
  float CanHandleEvent(const mitk::StateEvent* stateEvent) const;

  /// \brief Installs an event filter that can reject a state machine event or they let it pass through.
  virtual void InstallEventFilter(const MIDASEventFilter::Pointer eventFilter);

  /// \brief Removes an event filter that can reject a state machine event or they let it pass through.
  virtual void RemoveEventFilter(const MIDASEventFilter::Pointer eventFilter);

  /// \brief Gets the list of the installed event filters.
  std::vector<MIDASEventFilter::Pointer> GetEventFilters() const;

  /// \brief Tells if the event is rejected by the installed event filters or they let pass through.
  bool IsFiltered(const mitk::StateEvent* stateEvent) const;

protected:

  /// Tells if this state machine can handle the event. This function is called only if the event
  /// has not been rejected by one of the event filters.
  /// Derived classes should override this function in favor of mitk::StateMachine::CanHandleEvent
  /// if they can process also other kinds of events than what their base class can do.
  virtual float CanHandle(const mitk::StateEvent* stateEvent) const = 0;

private:

  /// \brief Filter the events that are sent to the interactors.
  std::vector<MIDASEventFilter::Pointer> m_EventFilters;

};

}

#endif

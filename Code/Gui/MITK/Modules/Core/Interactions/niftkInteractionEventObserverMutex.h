/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef InteractionEventObserverMutex_h
#define InteractionEventObserverMutex_h

#include "niftkCoreExports.h"

#include <unordered_map>

namespace mitk
{
class InteractionEventObserver;
}

namespace niftk
{

/// \brief Helper class to prevent unwanted simultaneous interactions.
///
/// This class helps to prevent processing multiple interactions by different
/// state machines at the same time.
///
/// This can happen for example if an interactor changes state when a mouse
/// button is pressed and goes back to the start state when the button is
/// released. If another interactor is active at the same time, it can
/// process key events while the mouse button is pressed and the first
/// interactor is doing something.
///
/// To prevent simultaneous interactions while someone interacts with the first
/// interactor, this interactor should get a lock of this mutex when the mouse
/// button pressed, and release the lock when the button is released.
/// When the lock of the mutex is acquired, it saves the actual state of the
/// currently registered interactors (whether they are enabled or disabled),
/// and disables them. When the lock is released, it restores the original state
/// of the interactors.

class NIFTKCORE_EXPORT InteractionEventObserverMutex
{
public:

  /// \brief Returns the single instance of this class.
  static InteractionEventObserverMutex* GetInstance();

  /// \brief Acquires the lock of the mutex.
  /// The lock has to be released when this function is called.
  /// \param guardedObserver
  ///     The observer that wants to be active exclusively.
  ///     The function is supposed to be called by the observer that
  ///     starts a longer interaction and wants to exclude other observers,
  ///     so the argument should be 'this'.
  void Lock(mitk::InteractionEventObserver* guardedObserver);

  /// \brief Releases the lock of the mutex.
  /// The lock has to be owned by guardedObserver when this function is called.
  /// \param guardedObserver
  ///     The observer that wanted to be active exclusively.
  ///     The function is supposed to be called by the observer that
  ///     starts a longer interaction and wants to exclude other observers,
  ///     so the argument should be 'this'.
  void Unlock(mitk::InteractionEventObserver* guardedObserver);

private:

  InteractionEventObserverMutex();

  ~InteractionEventObserverMutex();

  /// \brief Stores the current state of other interaction event observers and disables them.
  ///
  /// This is a utility function to disable every interaction event observer other than this
  /// one. Other interactions have to be disabled for the time while a longer, non-atomic
  /// mouse interaction is going on. E.g. erasing contour segments with the draw tool requires
  /// moving the mouse while keeping the middle mouse button pressed. Accidently hitting a key
  /// during this operation may trigger an action of the another state machine, although the
  /// two actions are not supposed to run simultaneously.
  ///
  /// For instance, if you press 'c' to clean a slice while you are erasing a contour segment
  /// by the mouse in the general segmentor, the erasure is interrupted and the draw tool
  /// state machine does not go back to its start state when you release the mouse button.
  ///
  /// This function saves the current state of all interaction event observers except this one
  /// (whether they are enabled or disabled) and it disables them. The observers can be put
  /// back in operation by ReactivateOtherInteractionEventObservers.
  ///
  /// This function should be invoked from Tool actions triggered by mouse press event where
  /// there is an other action triggered by mouse release event that brings the state machine
  /// back to the start state.
  void DeactivateOtherInteractionEventObservers();

  /// \brief Restores the state of other interaction event observers.
  ///
  /// It restores the state of every interaction event observer other than this one.
  /// The restored state is the one saved by the last DeactivateOtherInteractionEventObservers
  /// call. The function assumes that the list of interaction event observers has not changed
  ///
  /// This function should be invoked from Tool actions triggered by mouse release event where
  /// there was a previous action triggered by mouse press event during which the other
  /// interaction event observers have been disabled.
  void ReactivateOtherInteractionEventObservers();

  mitk::InteractionEventObserver* m_GuardedObserver;

  /// \brief Stores the state of every registered interaction event observer apart from this one.
  /// The state can be enabled (true) or disabled (false). The variable is managed by the
  std::unordered_map<mitk::InteractionEventObserver*, bool> m_StateOfOtherInteractionEventObservers;

};

}

#endif

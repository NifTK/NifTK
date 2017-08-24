/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPointSetDataInteractor_h
#define niftkPointSetDataInteractor_h

#include "niftkMIDASExports.h"

#include <vector>

#include <mitkPointSetDataInteractor.h>
#include <mitkInteractionEventObserver.h>

#include "niftkFilteringStateMachine.h"

namespace niftk
{
/// \class PointSetDataInteractor
/// \brief Derived from mitk::PointSetDataInteractor so we can 'round' positions to voxel centres.
/// \ingroup Interaction
class NIFTKMIDAS_EXPORT PointSetDataInteractor
  : public mitk::PointSetDataInteractor,
    public mitk::InteractionEventObserver,
    public FilteringStateMachine
{
public:
  mitkClassMacro(PointSetDataInteractor, mitk::PointSetDataInteractor)
  itkFactorylessNewMacro(Self)
  itkCloneMacro(Self)

protected:

  /// \brief Constructs a point set data interactor.
  PointSetDataInteractor();

  /// \brief Destructs the point set data interactor.
  virtual ~PointSetDataInteractor();

  /// \brief Notifies this interaction event observer about an interaction event.
  virtual void Notify(mitk::InteractionEvent* interactionEvent, bool isHandled) override;

  /// \brief Tells if this tool can handle the given event.
  ///
  /// This implementation delegates the call to FilteringStateMachine::CanHandleEvent(),
  /// that checks if the event is filtered by one of the installed event filters and if not,
  /// calls CanHandle() and returns with its result.
  ///
  /// Note that this function is purposefully not virtual. Eventual subclasses should
  /// override the CanHandle function.
  virtual bool FilterEvents(mitk::InteractionEvent* event, mitk::DataNode* dataNode) override;

  /// \brief Maps actions strings from the loaded state machine pattern to functions of the data interactor.
  /// These functions are called when an action from the state machine pattern is executed.
  virtual void ConnectActionsAndFunctions() override;

  /// Overrides mitk::PointSetDataInteractor::CheckCondition() so that for any operation the
  /// display position is modified to be in the middle of a voxel.
  virtual bool CheckCondition(const mitk::StateMachineCondition& condition, const mitk::InteractionEvent* event) override;

  /// Overrides mitk::PointSetDataInteractor::ExecuteAction() so that for any operation the
  /// display position is modified to be in the middle of a voxel.
  virtual bool ExecuteAction(mitk::StateMachineAction* action, mitk::InteractionEvent* event) override;

  /// \brief Disables other interactors while a point is being added, moved or points are removed.
  virtual bool Lock(mitk::StateMachineAction* action, mitk::InteractionEvent* event);

  /// \brief Re-enables other interactors after a point has been being added, moved or points have been being removed.
  virtual bool Unlock(mitk::StateMachineAction* action, mitk::InteractionEvent* event);

};

}

#endif

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkMIDASPointSetInteractor_h
#define mitkMIDASPointSetInteractor_h

#include "niftkMIDASExports.h"

#include <mitkPointSetDataInteractor.h>
#include "mitkMIDASStateMachine.h"

#include <vector>

namespace mitk
{
/**
 * \class mitkMIDASPointSetInteractor
 * \brief Derived from mitkPointSetInteractor so we can handle the mouse move event.
 * \ingroup Interaction
 */
class NIFTKMIDAS_EXPORT MIDASPointSetInteractor : public mitk::PointSetDataInteractor, public MIDASStateMachine
{
public:
  mitkClassMacro(MIDASPointSetInteractor, mitk::PointSetDataInteractor);
  itkFactorylessNewMacro(Self)
  itkCloneMacro(Self)

protected:
  /**
   * \brief Constructor with Param n for limited Set of Points
   *
   * If no n is set, then the number of points is unlimited
   * n=0 is not supported. In this case, n is set to 1.
   */
  MIDASPointSetInteractor();

  /**
   * \brief Default Destructor
   **/
  virtual ~MIDASPointSetInteractor();

  /// \brief Tells if this tool can handle the given event.
  ///
  /// This implementation delegates the call to mitk::MIDASStateMachine::CanHandleEvent(),
  /// that checks if the event is filtered by one of the installed event filters and if not,
  /// calls CanHandle() and returns with its result.
  ///
  /// Note that this function is purposefully not virtual. Eventual subclasses should
  /// override the CanHandle function.
  bool FilterEvents(mitk::InteractionEvent* event, mitk::DataNode* dataNode);

  virtual bool CanHandle(mitk::InteractionEvent* event);

  /**
  * @brief Convert the given Actions to Operations and send to data and UndoController
  *
  * Overrides mitk::PointSetInteractor::ExecuteAction() so that for any operation the
  * display position is modified to be in the middle of a pixel.
  */
  virtual bool ExecuteAction(mitk::StateMachineAction* action, mitk::InteractionEvent* event);

};
}
#endif

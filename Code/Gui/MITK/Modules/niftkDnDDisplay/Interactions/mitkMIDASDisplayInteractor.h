/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkMIDASDisplayInteractor_h
#define mitkMIDASDisplayInteractor_h

#include <niftkDnDDisplayExports.h>

#include <mitkDisplayInteractor.h>

#include <vector>

namespace mitk
{
  class BaseRenderer;
  class SliceNavigationController;

  /**
   *\class MIDASDisplayInteractor
   *@brief Observer that manages the interaction with the display.
   *
   * mitk::ToolManager reloads the configuration of the registered mitk::DisplayInteractor
   * objects when a tool is switched on. This configuration conflicts with most of the tools.
   *
   * @ingroup Interaction
   **/
  /**
   * Inherits from mitk::InteractionEventObserver since it doesn't alter any data (only their representation),
   * and its actions cannot be associated with a DataNode. Also inherits from EventStateMachine
   */
  class NIFTKDNDDISPLAY_EXPORT MIDASDisplayInteractor: public DisplayInteractor
  {
  public:
    mitkClassMacro(MIDASDisplayInteractor, DisplayInteractor)
    // MIDAS customisation: renderers and slice navigation controllers passed as arguments
    mitkNewMacro1Param(Self, const std::vector<mitk::BaseRenderer*>&);

    /**
     * By this function the Observer gets notifier about new events.
     * Here it is adapted to pass the events to the state machine in order to use
     * its infrastructure.
     * It also checks if event is to be accepted when i already has been processed by a DataInteractor.
     */
    virtual void Notify(InteractionEvent* interactionEvent, bool isHandled);

  protected:
    MIDASDisplayInteractor(const std::vector<mitk::BaseRenderer*>& renderers);
    virtual ~MIDASDisplayInteractor();

    virtual void ConnectActionsAndFunctions();

    virtual bool InitZoom(StateMachineAction*, InteractionEvent*);

  private:
    /**
     * Renderers of the MIDASStdMultiWidget that this display interactor belongs to.
     */
    std::vector<mitk::BaseRenderer*> m_Renderers;

    /**
     * Slice navigation controllers of the MIDASStdMultiWidget that this display interactor belongs to.
     */
    mitk::SliceNavigationController* m_SliceNavigationControllers[3];
  };
}
#endif

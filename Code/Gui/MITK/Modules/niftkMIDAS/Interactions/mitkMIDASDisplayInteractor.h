/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitk_MIDASDisplayInteractor_h
#define mitk_MIDASDisplayInteractor_h

#include <niftkMIDASExports.h>

#include <mitkInteractionEventObserver.h>
#include <mitkCommon.h>

#include <vector>

namespace mitk
{
  class BaseRenderer;
  class SliceNavigationController;

  /**
   *\class MIDASDisplayInteractor
   *@brief Observer that manages the interaction with the display.
   *
   * This includes the interaction of Zooming, Panning, Scrolling and adjusting the LevelWindow.
   *
   * This class is basically a copy of mitk::DisplayInteractor. The class is deliberetaly not a
   * derived from mitk::DisplayInteractor because of an unresolved MITK bug:
   *
   *   http://bugs.mitk.org/show_bug.cgi?id=14829
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
  class NIFTKMIDAS_EXPORT MIDASDisplayInteractor: public EventStateMachine, public InteractionEventObserver
  {
  public:
    mitkClassMacro(MIDASDisplayInteractor, EventStateMachine)
    // MIDAS customisation: renderers and slice navigation controllers passed as arguments
    mitkNewMacro2Param(Self, const std::vector<mitk::BaseRenderer*>&, const std::vector<mitk::SliceNavigationController*>&);

    /**
     * By this function the Observer gets notifier about new events.
     * Here it is adapted to pass the events to the state machine in order to use
     * its infrastructure.
     * It also checks if event is to be accepted when i already has been processed by a DataInteractor.
     */
    virtual void Notify(InteractionEvent* interactionEvent, bool isHandled);

  protected:
    // MIDAS customisation: slice navigation controllers passed as arguments
    MIDASDisplayInteractor(const std::vector<mitk::BaseRenderer*>& renderers, const std::vector<mitk::SliceNavigationController*>& sliceNavigationControllers);
    virtual ~MIDASDisplayInteractor();
    /**
     * Derived function.
     * Connects the action names used in the state machine pattern with functions implemented within
     * this InteractionEventObserver. This is only necessary here because the events are processed by the state machine.
     */
    void ConnectActionsAndFunctions();
    /**
     * Derived function.
     * Is executed when config object is set / changed.
     * Here it is used to read out the parameters set in the configuration file,
     * and set the member variables accordingly.
     */
    virtual void ConfigurationChanged();

    /**
     * Derived function.
     * Is executed when config object is set / changed.
     * Here it is used to read out the parameters set in the configuration file,
     * and set the member variables accordingly.
     */
    virtual bool FilterEvents(InteractionEvent* interactionEvent, DataNode* dataNode);

    /**
     * \brief Initializes an interaction, saves the pointers start position for further reference.
     */
    virtual bool Init(StateMachineAction*, InteractionEvent*);
    /**
     * \brief Initializes an interaction, saves the pointers start position for further reference.
     */
    virtual bool InitZoom(StateMachineAction*, InteractionEvent*);
    /**
     * \brief Performs panning of the data set in the render window.
     */
    virtual bool Move(StateMachineAction*, InteractionEvent*);
    /**
     * \brief Performs zooming relative to mouse/pointer movement.
     *
     * Behavior is determined by \see m_ZoomDirection and \see m_ZoomFactor.
     *
     */
    virtual bool Zoom(StateMachineAction*, InteractionEvent*);
    /**
     * \brief Performs scrolling relative to mouse/pointer movement.
     *
     * Behavior is determined by \see m_ScrollDirection and \see m_AutoRepeat.
     *
     */
    virtual bool Scroll(StateMachineAction*, InteractionEvent*);
    /**
     * \brief Scrolls one layer up
     */
    virtual bool ScrollOneDown(StateMachineAction*, InteractionEvent*);
    /**
     * \brief Scrolls one layer down
     */
    virtual bool ScrollOneUp(StateMachineAction*, InteractionEvent*);
    /**
     * \brief Adjusts the level windows relative to mouse/pointer movement.
     */
    virtual bool AdjustLevelWindow(StateMachineAction*, InteractionEvent*);

  private:
    /**
     * \brief Coordinate of the pointer at begin of an interaction
     */
    mitk::Point2D m_StartDisplayCoordinate;
    /**
     * \brief Coordinate of the pointer at begin of an interaction translated to mm unit
     */
    mitk::Point2D m_StartCoordinateInMM;
    /**
     * \brief Coordinate of the pointer in the last step within an interaction.
     */
    mitk::Point2D m_LastDisplayCoordinate;
    /**
     * \brief Current coordinates of the pointer.
     */
    mitk::Point2D m_CurrentDisplayCoordinate;
    /**
     * \brief Modifier that defines how many slices are scrolled per pixel that the mouse has moved
     *
     * This modifier defines how many slices the scene is scrolled per pixel that the mouse cursor has moved.
     * By default the modifier is 4. This means that when the user moves the cursor by 4 pixels in Y-direction
     * the scene is scrolled by one slice. If the user has moved the the cursor by 20 pixels, the scene is
     * scrolled by 5 slices.
     *
     * If the cursor has moved less than m_IndexToSliceModifier pixels the scene is scrolled by one slice.
     */
    int m_IndexToSliceModifier;

    /** Defines behavior at end of data set.
     *  If set to true it will restart at end of data set from the beginning.
     */
    bool m_AutoRepeat;

    /**
     * Defines scroll behavior.
     * Default is up/down movement of pointer performs scrolling
     */
    std::string m_ScrollDirection;

    /**
     * Defines scroll behavior.
     * Default is up/down movement of pointer performs zooming
     */
    std::string m_ZoomDirection;

    /**
     * Determines if the Observer reacts to events that already have been processed by a DataInteractor.
     * The default value is false.
     */
    bool m_AlwaysReact;

    /**
     * Factor to adjust zooming speed.
     */
    float m_ZoomFactor;

    // MIDAS customisation: slice navigation controllers for zooming around the crosshair
    /**
     * Renderers of the MIDASStdMultiWidget that this display interactor belongs to.
     */
    std::vector<mitk::BaseRenderer*> m_Renderers;

    /**
     * Slice navigation controllers of the MIDASStdMultiWidget that this display interactor belongs to.
     */
    std::vector<mitk::SliceNavigationController*> m_SliceNavigationControllers;
  };
}
#endif

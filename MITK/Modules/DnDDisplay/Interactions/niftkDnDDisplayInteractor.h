/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkDnDDisplayInteractor_h
#define niftkDnDDisplayInteractor_h

#include <niftkDnDDisplayExports.h>

#include <vector>

#include <QObject>

#include <mitkDisplayInteractor.h>

#include <niftkDnDDisplayEnums.h>

class QmitkRenderWindow;
class QTimer;

namespace mitk
{
class FocusManager;
}


namespace niftk
{

class SingleViewerWidget;

/**
 *\class DnDDisplayInteractor
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
class NIFTKDNDDISPLAY_EXPORT DnDDisplayInteractor: public QObject, public mitk::DisplayInteractor
{
  Q_OBJECT

public:
  mitkClassMacro(DnDDisplayInteractor, DisplayInteractor)
  mitkNewMacro1Param(Self, SingleViewerWidget*)

  /**
   * By this function the Observer gets notifier about new events.
   * Here it is adapted to pass the events to the state machine in order to use
   * its infrastructure.
   * It also checks if event is to be accepted when i already has been processed by a DataInteractor.
   */
  virtual void Notify(mitk::InteractionEvent* interactionEvent, bool isHandled) override;

protected:
  DnDDisplayInteractor(SingleViewerWidget* viewer);
  virtual ~DnDDisplayInteractor();

  virtual void ConnectActionsAndFunctions() override;

  virtual bool StartSelectingPosition(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  virtual bool SelectPosition(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  virtual bool StopSelectingPosition(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Like Superclass::Init, but blocks the update and selects the focused window.
  virtual bool StartPanning(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  virtual bool Pan(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief
  virtual bool StopPanning(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Like Superclass::Init, but blocks the update and selects the focused window.
  /// It also changes the selected position to the middle of the focused voxel.
  virtual bool StartZooming(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  virtual bool Zoom(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent) override;

  /// \brief
  virtual bool StopZooming(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Switches to axial window layout.
  virtual bool SetWindowLayoutToAxial(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Switches to sagittal window layout.
  virtual bool SetWindowLayoutToSagittal(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Switches to coronal window layout.
  virtual bool SetWindowLayoutToCoronal(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Switches to 3D window layout.
  virtual bool SetWindowLayoutTo3D(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Switches to multi window layout.
  virtual bool SetWindowLayoutToMulti(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Toggles between single and multi window layout.
  virtual bool ToggleMultiWindowLayout(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Toggles the visibility of the cursor.
  virtual bool ToggleCursorVisibility(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Toggles displaying the direction annotations on/off.
  virtual bool ToggleDirectionAnnotations(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Toggles displaying the intensity annotation on/off.
  virtual bool TogglePositionAnnotation(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Toggles displaying the intensity annotation on/off.
  virtual bool ToggleIntensityAnnotation(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Toggles displaying the intensity annotation on/off.
  virtual bool TogglePropertyAnnotation(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Selects the previous slice.
  /// The slices are ordered in the following way:
  ///   <li>axial: inferior to superior
  ///   <li>sagittal: right to left
  ///   <li>coronal: anterior to posterior
  virtual bool SelectPreviousSlice(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Selects the next slice.
  /// The slices are ordered in the following way:
  ///   <li>axial: inferior to superior
  ///   <li>sagittal: right to left
  ///   <li>coronal: anterior to posterior
  virtual bool SelectNextSlice(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Selects the previous time step.
  virtual bool SelectPreviousTimeStep(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Selects the next time step.
  virtual bool SelectNextTimeStep(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Starts scrolling through slices in a loop backwards.
  virtual bool StartScrollingThroughSlicesBackwards(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Starts scrolling through slices in a loop in posterior direction.
  virtual bool StartScrollingThroughSlicesForwards(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Starts scrolling through time steps in a loop, backwards.
  virtual bool StartScrollingThroughTimeStepsBackwards(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Starts scrolling through time steps in a loop, forwards.
  virtual bool StartScrollingThroughTimeStepsForwards(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

  /// \brief Stops scrolling through slices.
  virtual bool StopScrolling(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent);

private slots:

  /// \brief Selects the previous slice or the last slice if the first slice is currently selected.
  /// This slot connected to a timer when the user starts auto-scrolling slices backwards.
  void SelectPreviousSlice();

  /// \brief Selects the next slice or the first slice if the last slice is currently selected.
  /// This slot connected to a timer when the user starts auto-scrolling slices forwards.
  void SelectNextSlice();

  /// \brief Selects the previous time step or the last time step if the first time step is currently selected.
  /// This slot connected to a timer when the user starts auto-scrolling time steps backwards.
  void SelectPreviousTimeStep();

  /// \brief Selects the next time step or the first time step if the last time step is currently selected.
  /// This slot connected to a timer when the user starts auto-scrolling time steps forwards.
  void SelectNextTimeStep();

private:

  QmitkRenderWindow* GetRenderWindow(mitk::BaseRenderer* renderer);

  SingleViewerWidget* m_Viewer;

  std::vector<mitk::BaseRenderer*> m_Renderers;

  mitk::FocusManager* m_FocusManager;

  QTimer* m_AutoScrollTimer;

};

}

#endif

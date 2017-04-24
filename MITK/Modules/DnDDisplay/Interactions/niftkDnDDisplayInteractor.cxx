/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkDnDDisplayInteractor.h"

#include <string.h>

#include <QTimer>

#include <mitkBaseRenderer.h>
#include <mitkGlobalInteraction.h>
#include <mitkInteractionPositionEvent.h>
#include <mitkLine.h>
#include <mitkSliceNavigationController.h>

#include <niftkInteractionEventObserverMutex.h>

#include "niftkSingleViewerWidget.h"


namespace niftk
{

//-----------------------------------------------------------------------------
DnDDisplayInteractor::DnDDisplayInteractor(SingleViewerWidget* viewer)
: mitk::DisplayInteractor()
, m_Viewer(viewer)
, m_Renderers(4)
, m_FocusManager(mitk::GlobalInteraction::GetInstance()->GetFocusManager())
, m_AutoScrollTimer(new QTimer(this))
{
  const std::vector<QmitkRenderWindow*>& renderWindows = m_Viewer->GetRenderWindows();
  m_Renderers[0] = renderWindows[0]->GetRenderer();
  m_Renderers[1] = renderWindows[1]->GetRenderer();
  m_Renderers[2] = renderWindows[2]->GetRenderer();
  m_Renderers[3] = renderWindows[3]->GetRenderer();

  m_AutoScrollTimer->setInterval(200);
}


//-----------------------------------------------------------------------------
DnDDisplayInteractor::~DnDDisplayInteractor()
{
}


//-----------------------------------------------------------------------------
void DnDDisplayInteractor::Notify(mitk::InteractionEvent* interactionEvent, bool isHandled)
{
  mitk::BaseRenderer* renderer = interactionEvent->GetSender();
  if (std::find(m_Renderers.begin(), m_Renderers.end(), renderer) != m_Renderers.end())
  {
    Superclass::Notify(interactionEvent, isHandled);
  }
}


//-----------------------------------------------------------------------------
void DnDDisplayInteractor::ConnectActionsAndFunctions()
{
  /// Note:
  /// We do not call the overridden function here. It assign handlers to actions
  /// that are not defined for this state machine.

  CONNECT_FUNCTION("startSelectingPosition", StartSelectingPosition);
  CONNECT_FUNCTION("selectPosition", SelectPosition);
  CONNECT_FUNCTION("stopSelectingPosition", StopSelectingPosition);
  CONNECT_FUNCTION("startPanning", StartPanning);
  CONNECT_FUNCTION("pan", Pan);
  CONNECT_FUNCTION("stopPanning", StopPanning);
  CONNECT_FUNCTION("startZooming", StartZooming);
  CONNECT_FUNCTION("zoom", Zoom);
  CONNECT_FUNCTION("stopZooming", StopZooming);
  CONNECT_FUNCTION("setWindowLayoutToAxial", SetWindowLayoutToAxial);
  CONNECT_FUNCTION("setWindowLayoutToSagittal", SetWindowLayoutToSagittal);
  CONNECT_FUNCTION("setWindowLayoutToCoronal", SetWindowLayoutToCoronal);
  CONNECT_FUNCTION("setWindowLayoutTo3D", SetWindowLayoutTo3D);
  CONNECT_FUNCTION("setWindowLayoutToMulti", SetWindowLayoutToMulti);
  CONNECT_FUNCTION("toggleMultiWindowLayout", ToggleMultiWindowLayout);
  CONNECT_FUNCTION("selectPreviousWindow", SelectPreviousWindow);
  CONNECT_FUNCTION("selectNextWindow", SelectNextWindow);
  CONNECT_FUNCTION("selectAxialWindow", SelectAxialWindow);
  CONNECT_FUNCTION("selectSagittalWindow", SelectSagittalWindow);
  CONNECT_FUNCTION("selectCoronalWindow", SelectCoronalWindow);
  CONNECT_FUNCTION("select3DWindow", Select3DWindow);
  CONNECT_FUNCTION("selectPreviousViewer", SelectPreviousViewer);
  CONNECT_FUNCTION("selectNextViewer", SelectNextViewer);
  CONNECT_FUNCTION("selectViewer0", SelectViewer0);
  CONNECT_FUNCTION("selectViewer1", SelectViewer1);
  CONNECT_FUNCTION("selectViewer2", SelectViewer2);
  CONNECT_FUNCTION("selectViewer3", SelectViewer3);
  CONNECT_FUNCTION("selectViewer4", SelectViewer4);
  CONNECT_FUNCTION("selectViewer5", SelectViewer5);
  CONNECT_FUNCTION("selectViewer6", SelectViewer6);
  CONNECT_FUNCTION("selectViewer7", SelectViewer7);
  CONNECT_FUNCTION("selectViewer8", SelectViewer8);
  CONNECT_FUNCTION("selectViewer9", SelectViewer9);
  CONNECT_FUNCTION("toggleCursorVisibility", ToggleCursorVisibility);
  CONNECT_FUNCTION("toggleDirectionAnnotations", ToggleDirectionAnnotations);
  CONNECT_FUNCTION("togglePositionAnnotation", TogglePositionAnnotation);
  CONNECT_FUNCTION("toggleIntensityAnnotation", ToggleIntensityAnnotation);
  CONNECT_FUNCTION("togglePropertyAnnotation", TogglePropertyAnnotation);

  CONNECT_FUNCTION("selectVoxelOnLeft", SelectVoxelOnLeft);
  CONNECT_FUNCTION("selectVoxelOnRight", SelectVoxelOnRight);
  CONNECT_FUNCTION("selectVoxelAbove", SelectVoxelAbove);
  CONNECT_FUNCTION("selectVoxelBelow", SelectVoxelBelow);
  CONNECT_FUNCTION("selectPreviousSlice", SelectPreviousSlice);
  CONNECT_FUNCTION("selectNextSlice", SelectNextSlice);
  CONNECT_FUNCTION("selectPreviousTimeStep", SelectPreviousTimeStep);
  CONNECT_FUNCTION("selectNextTimeStep", SelectNextTimeStep);

  CONNECT_FUNCTION("startScrollingThroughSlicesBackwards", StartScrollingThroughSlicesBackwards);
  CONNECT_FUNCTION("startScrollingThroughSlicesForwards", StartScrollingThroughSlicesForwards);
  CONNECT_FUNCTION("startScrollingThroughTimeStepsBackwards", StartScrollingThroughTimeStepsBackwards);
  CONNECT_FUNCTION("startScrollingThroughTimeStepsForwards", StartScrollingThroughTimeStepsForwards);
  CONNECT_FUNCTION("stopScrolling", StopScrolling);
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* DnDDisplayInteractor::GetRenderWindow(mitk::BaseRenderer* renderer)
{
  QmitkRenderWindow* renderWindow = 0;

  std::size_t i = std::find(m_Renderers.begin(), m_Renderers.end(), renderer) - m_Renderers.begin();

  if (i < 4)
  {
    renderWindow = m_Viewer->GetRenderWindows()[i];
  }

  return renderWindow;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::StartSelectingPosition(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  niftk::InteractionEventObserverMutex::GetInstance()->Lock(this);
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectPosition(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  mitk::BaseRenderer* renderer = interactionEvent->GetSender();

  /// Note:
  /// We do not re-implement position selection for the 3D window.
  if (renderer == m_Renderers[3])
  {
    return false;
  }

  mitk::InteractionPositionEvent* positionEvent = dynamic_cast<mitk::InteractionPositionEvent*>(interactionEvent);
  assert(positionEvent);

  // First, check if the slice navigation controllers have a valid geometry,
  // i.e. an image is loaded.
  if (!m_Renderers[0]->GetSliceNavigationController()->GetCreatedWorldGeometry())
  {
    return false;
  }

  bool updateWasBlocked = m_Viewer->BlockUpdate(true);

  if (renderer != m_FocusManager->GetFocused())
  {
    QmitkRenderWindow* renderWindow = this->GetRenderWindow(renderer);
    m_Viewer->SetSelectedRenderWindow(renderWindow);
    m_Viewer->SetFocused();
  }

  // Selects the point under the mouse pointer in the slice navigation controllers.
  // In the MultiWindowWidget this puts the crosshair to the mouse position, and
  // selects the slice in the two other render window.
  const mitk::Point3D& positionInWorld = positionEvent->GetPositionInWorld();
  m_Viewer->SetSelectedPosition(positionInWorld);

  m_Viewer->BlockUpdate(updateWasBlocked);

  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::StopSelectingPosition(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  niftk::InteractionEventObserverMutex::GetInstance()->Unlock(this);
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::StartPanning(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  niftk::InteractionEventObserverMutex::GetInstance()->Lock(this);

  mitk::InteractionPositionEvent* positionEvent = dynamic_cast<mitk::InteractionPositionEvent*>(interactionEvent);
  assert(positionEvent);

  // First, check if the slice navigation controllers have a valid geometry,
  // i.e. an image is loaded.
  if (!m_Renderers[0]->GetSliceNavigationController()->GetCreatedWorldGeometry())
  {
    return false;
  }

  bool updateWasBlocked = m_Viewer->BlockUpdate(true);

  mitk::BaseRenderer* renderer = interactionEvent->GetSender();
  if (renderer != m_FocusManager->GetFocused())
  {
    QmitkRenderWindow* renderWindow = this->GetRenderWindow(renderer);
    m_Viewer->SetSelectedRenderWindow(renderWindow);
    m_Viewer->SetFocused();
  }

  bool result = this->Init(action, interactionEvent);

  m_Viewer->BlockUpdate(updateWasBlocked);

  return result;
}


////-----------------------------------------------------------------------------
bool DnDDisplayInteractor::Pan(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  return Superclass::Move(action, interactionEvent);
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::StopPanning(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  niftk::InteractionEventObserverMutex::GetInstance()->Unlock(this);
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::StartZooming(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  niftk::InteractionEventObserverMutex::GetInstance()->Lock(this);

  mitk::InteractionPositionEvent* positionEvent = dynamic_cast<mitk::InteractionPositionEvent*>(interactionEvent);
  assert(positionEvent);

  // First, check if the slice navigation controllers have a valid geometry,
  // i.e. an image is loaded.
  if (!m_Renderers[0]->GetSliceNavigationController()->GetCreatedWorldGeometry())
  {
    return false;
  }

  bool updateWasBlocked = m_Viewer->BlockUpdate(true);

  mitk::BaseRenderer* renderer = interactionEvent->GetSender();
  if (renderer != m_FocusManager->GetFocused())
  {
    QmitkRenderWindow* renderWindow = this->GetRenderWindow(renderer);
    m_Viewer->SetSelectedRenderWindow(renderWindow);
    m_Viewer->SetFocused();
  }

  /// Note that the zoom focus must always be the selected position,
  /// i.e. the position at the cursor (crosshair).
  mitk::Point3D focusPoint3DInMm = m_Viewer->GetSelectedPosition();

  mitk::Point2D focusPoint2DInMm;
  mitk::Point2D focusPoint2DInPx;
  mitk::Point2D focusPoint2DInPxUL;

  mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();
  displayGeometry->Map(focusPoint3DInMm, focusPoint2DInMm);
  displayGeometry->WorldToDisplay(focusPoint2DInMm, focusPoint2DInPx);
  displayGeometry->DisplayToULDisplay(focusPoint2DInPx, focusPoint2DInPxUL);

  // Create a new position event with the selected position.
  mitk::InteractionPositionEvent::Pointer positionEvent2 = mitk::InteractionPositionEvent::New(renderer, focusPoint2DInPxUL, focusPoint3DInMm);

  bool result = this->Init(action, positionEvent2);

  m_Viewer->BlockUpdate(updateWasBlocked);

  return result;
}


////-----------------------------------------------------------------------------
bool DnDDisplayInteractor::Zoom(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  return Superclass::Zoom(action, interactionEvent);
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::StopZooming(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  niftk::InteractionEventObserverMutex::GetInstance()->Unlock(this);
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SetWindowLayoutToAxial(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  m_Viewer->SetWindowLayout(WINDOW_LAYOUT_AXIAL);
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SetWindowLayoutToSagittal(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  m_Viewer->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SetWindowLayoutToCoronal(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  m_Viewer->SetWindowLayout(WINDOW_LAYOUT_CORONAL);
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SetWindowLayoutTo3D(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  m_Viewer->SetWindowLayout(WINDOW_LAYOUT_3D);
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SetWindowLayoutToMulti(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  m_Viewer->ToggleMultiWindowLayout();
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::ToggleMultiWindowLayout(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  m_Viewer->ToggleMultiWindowLayout();
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectPreviousWindow(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  if (auto previousWindow = m_Viewer->GetPreviousWindow())
  {
    m_Viewer->SetSelectedRenderWindow(previousWindow);
  }
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectNextWindow(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  if (auto nextWindow = m_Viewer->GetNextWindow())
  {
    m_Viewer->SetSelectedRenderWindow(nextWindow);
  }
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectAxialWindow(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  QmitkRenderWindow* window = m_Viewer->GetAxialWindow();
  if (window && window->isVisible())
  {
    m_Viewer->SetSelectedRenderWindow(window);
  }

  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectSagittalWindow(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  QmitkRenderWindow* window = m_Viewer->GetSagittalWindow();
  if (window && window->isVisible())
  {
    m_Viewer->SetSelectedRenderWindow(window);
  }

  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectCoronalWindow(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  QmitkRenderWindow* window = m_Viewer->GetCoronalWindow();
  if (window && window->isVisible())
  {
    m_Viewer->SetSelectedRenderWindow(window);
  }

  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::Select3DWindow(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  QmitkRenderWindow* window = m_Viewer->Get3DWindow();
  if (window && window->isVisible())
  {
    m_Viewer->SetSelectedRenderWindow(window);
  }

  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectPreviousViewer(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  emit m_Viewer->SelectPreviousViewer();
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectNextViewer(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  emit m_Viewer->SelectNextViewer();
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectViewer0(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  emit m_Viewer->SelectViewer(0);
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectViewer1(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  emit m_Viewer->SelectViewer(1);
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectViewer2(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  emit m_Viewer->SelectViewer(2);
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectViewer3(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  emit m_Viewer->SelectViewer(3);
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectViewer4(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  emit m_Viewer->SelectViewer(4);
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectViewer5(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  emit m_Viewer->SelectViewer(5);
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectViewer6(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  emit m_Viewer->SelectViewer(6);
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectViewer7(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  emit m_Viewer->SelectViewer(7);
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectViewer8(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  emit m_Viewer->SelectViewer(8);
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectViewer9(mitk::StateMachineAction* /*action*/, mitk::InteractionEvent* /*interactionEvent*/)
{
  emit m_Viewer->SelectViewer(9);
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::ToggleCursorVisibility(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  m_Viewer->ToggleCursorVisibility();
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::ToggleDirectionAnnotations(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  m_Viewer->ToggleDirectionAnnotations();
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::TogglePositionAnnotation(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  m_Viewer->TogglePositionAnnotation();
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::ToggleIntensityAnnotation(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  m_Viewer->ToggleIntensityAnnotation();
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::TogglePropertyAnnotation(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  m_Viewer->TogglePropertyAnnotation();
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectVoxelOnLeft(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  mitk::BaseRenderer* renderer = interactionEvent->GetSender();

  /// Note:
  /// We do not selecting voxels for the 3D window.
  if (renderer == m_Renderers[3])
  {
    return true;
  }

  WindowOrientation focusedOrientation = m_Viewer->GetOrientation();

  WindowOrientation navigationOrientation;
  int delta;
  if (focusedOrientation == WINDOW_ORIENTATION_SAGITTAL)
  {
    navigationOrientation = WINDOW_ORIENTATION_CORONAL;
    delta = +1;
  }
  else // if axial or coronal
  {
    navigationOrientation = WINDOW_ORIENTATION_SAGITTAL;
    delta = -1;
  }

  m_Viewer->MoveSlice(navigationOrientation, delta);

  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectVoxelOnRight(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  mitk::BaseRenderer* renderer = interactionEvent->GetSender();

  /// Note:
  /// We do not selecting voxels for the 3D window.
  if (renderer == m_Renderers[3])
  {
    return true;
  }

  WindowOrientation focusedOrientation = m_Viewer->GetOrientation();

  WindowOrientation navigationOrientation;
  int delta;
  if (focusedOrientation == WINDOW_ORIENTATION_SAGITTAL)
  {
    navigationOrientation = WINDOW_ORIENTATION_CORONAL;
    delta = -1;
  }
  else // if axial or coronal
  {
    navigationOrientation = WINDOW_ORIENTATION_SAGITTAL;
    delta = +1;
  }

  m_Viewer->MoveSlice(navigationOrientation, delta);

  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectVoxelAbove(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  mitk::BaseRenderer* renderer = interactionEvent->GetSender();

  /// Note:
  /// We do not selecting voxels for the 3D window.
  if (renderer == m_Renderers[3])
  {
    return true;
  }

  WindowOrientation focusedOrientation = m_Viewer->GetOrientation();

  WindowOrientation navigationOrientation;
  int delta;
  if (focusedOrientation == WINDOW_ORIENTATION_AXIAL)
  {
    navigationOrientation = WINDOW_ORIENTATION_CORONAL;
    delta = +1;
  }
  else // if sagittal or coronal
  {
    navigationOrientation = WINDOW_ORIENTATION_AXIAL;
    delta = -1;
  }

  m_Viewer->MoveSlice(navigationOrientation, delta);

  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectVoxelBelow(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  mitk::BaseRenderer* renderer = interactionEvent->GetSender();

  /// Note:
  /// We do not selecting voxels for the 3D window.
  if (renderer == m_Renderers[3])
  {
    return true;
  }

  WindowOrientation focusedOrientation = m_Viewer->GetOrientation();

  WindowOrientation navigationOrientation;
  int delta;
  if (focusedOrientation == WINDOW_ORIENTATION_AXIAL)
  {
    navigationOrientation = WINDOW_ORIENTATION_CORONAL;
    delta = -1;
  }
  else // if sagittal or coronal
  {
    navigationOrientation = WINDOW_ORIENTATION_AXIAL;
    delta = +1;
  }

  m_Viewer->MoveSlice(navigationOrientation, delta);

  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectPreviousSlice(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  mitk::BaseRenderer* renderer = interactionEvent->GetSender();

  /// Note:
  /// We do not implement scrolling for the 3D window.
  if (renderer == m_Renderers[3])
  {
    return true;
  }

  bool updateWasBlocked = m_Viewer->BlockUpdate(true);

  if (renderer != m_FocusManager->GetFocused())
  {
    QmitkRenderWindow* renderWindow = this->GetRenderWindow(renderer);
    m_Viewer->SetSelectedRenderWindow(renderWindow);
    m_Viewer->SetFocused();
  }

  /// Note:
  /// This does not work if the slice are locked.
  /// See:
  ///   SingleViewerWidget::SetNavigationControllerEventListening(bool)
  /// and
  ///   QmitkMultiWindowWidget::SetWidgetPlanesLocked(bool)

//  bool result = Superclass::ScrollOneUp(action, interactionEvent);

  WindowOrientation orientation = m_Viewer->GetOrientation();
  m_Viewer->MoveSlice(orientation, -1);

  m_Viewer->BlockUpdate(updateWasBlocked);

//  return result;
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectNextSlice(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  mitk::BaseRenderer* renderer = interactionEvent->GetSender();

  /// Note:
  /// We do not implement scrolling for the 3D window.
  if (renderer == m_Renderers[3])
  {
    return true;
  }

  bool updateWasBlocked = m_Viewer->BlockUpdate(true);

  if (renderer != m_FocusManager->GetFocused())
  {
    QmitkRenderWindow* renderWindow = this->GetRenderWindow(renderer);
    m_Viewer->SetSelectedRenderWindow(renderWindow);
    m_Viewer->SetFocused();
  }

  /// Note:
  /// This does not work if the slice are locked.
  /// See:
  ///   SingleViewerWidget::SetNavigationControllerEventListening(bool)
  /// and
  ///   QmitkMultiWindowWidget::SetWidgetPlanesLocked(bool)

//  bool result = Superclass::ScrollOneDown(action, interactionEvent);

  WindowOrientation orientation = m_Viewer->GetOrientation();
  m_Viewer->MoveSlice(orientation, +1);

  m_Viewer->BlockUpdate(updateWasBlocked);

//  return result;
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectPreviousTimeStep(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  int timeStep = m_Viewer->GetTimeStep() - 1;

  if (timeStep >= 0)
  {
    m_Viewer->SetTimeStep(timeStep);
  }

  return timeStep == m_Viewer->GetTimeStep();
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::SelectNextTimeStep(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  int timeStep = m_Viewer->GetTimeStep() + 1;

  if (timeStep <= m_Viewer->GetMaxTimeStep())
  {
    m_Viewer->SetTimeStep(timeStep);
  }

  return timeStep == m_Viewer->GetTimeStep();
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::StartScrollingThroughSlicesBackwards(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  niftk::InteractionEventObserverMutex::GetInstance()->Lock(this);

  this->connect(m_AutoScrollTimer, SIGNAL(timeout()), SLOT(SelectPreviousSlice()));
  m_AutoScrollTimer->start();

  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::StartScrollingThroughSlicesForwards(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  niftk::InteractionEventObserverMutex::GetInstance()->Lock(this);

  this->connect(m_AutoScrollTimer, SIGNAL(timeout()), SLOT(SelectNextSlice()));
  m_AutoScrollTimer->start();
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::StartScrollingThroughTimeStepsBackwards(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  niftk::InteractionEventObserverMutex::GetInstance()->Lock(this);

  this->connect(m_AutoScrollTimer, SIGNAL(timeout()), SLOT(SelectPreviousTimeStep()));
  m_AutoScrollTimer->start();
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::StartScrollingThroughTimeStepsForwards(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  niftk::InteractionEventObserverMutex::GetInstance()->Lock(this);

  this->connect(m_AutoScrollTimer, SIGNAL(timeout()), SLOT(SelectNextTimeStep()));
  m_AutoScrollTimer->start();
  return true;
}


//-----------------------------------------------------------------------------
bool DnDDisplayInteractor::StopScrolling(mitk::StateMachineAction* action, mitk::InteractionEvent* interactionEvent)
{
  niftk::InteractionEventObserverMutex::GetInstance()->Unlock(this);

  m_AutoScrollTimer->stop();
  m_AutoScrollTimer->disconnect(this);
  return true;
}


//-----------------------------------------------------------------------------
void DnDDisplayInteractor::SelectPreviousSlice()
{
  WindowOrientation orientation = m_Viewer->GetOrientation();
  m_Viewer->MoveSlice(orientation, -1, true);
}


//-----------------------------------------------------------------------------
void DnDDisplayInteractor::SelectNextSlice()
{
  WindowOrientation orientation = m_Viewer->GetOrientation();
  m_Viewer->MoveSlice(orientation, +1, true);
}


//-----------------------------------------------------------------------------
void DnDDisplayInteractor::SelectPreviousTimeStep()
{
  int timeStep = m_Viewer->GetTimeStep() - 1;
  if (timeStep < 0)
  {
    timeStep = m_Viewer->GetMaxTimeStep();
  }
  m_Viewer->SetTimeStep(timeStep);
}


//-----------------------------------------------------------------------------
void DnDDisplayInteractor::SelectNextTimeStep()
{
  int timeStep = m_Viewer->GetTimeStep() + 1;
  if (timeStep > m_Viewer->GetMaxTimeStep())
  {
    timeStep = 0;
  }
  m_Viewer->SetTimeStep(timeStep);
}

}

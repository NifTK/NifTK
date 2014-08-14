/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkDnDDisplayInteractor.h"

#include <string.h>

#include "../niftkSingleViewerWidget.h"

#include <mitkBaseRenderer.h>
#include <mitkGlobalInteraction.h>
#include <mitkInteractionPositionEvent.h>
#include <mitkLine.h>
#include <mitkSliceNavigationController.h>


//-----------------------------------------------------------------------------
mitk::DnDDisplayInteractor::DnDDisplayInteractor(niftkSingleViewerWidget* viewer)
: mitk::DisplayInteractor()
, m_Viewer(viewer)
, m_Renderers(4)
, m_FocusManager(mitk::GlobalInteraction::GetInstance()->GetFocusManager())
{
  const std::vector<QmitkRenderWindow*>& renderWindows = m_Viewer->GetRenderWindows();
  m_Renderers[0] = renderWindows[0]->GetRenderer();
  m_Renderers[1] = renderWindows[1]->GetRenderer();
  m_Renderers[2] = renderWindows[2]->GetRenderer();
  m_Renderers[3] = renderWindows[3]->GetRenderer();
}


//-----------------------------------------------------------------------------
mitk::DnDDisplayInteractor::~DnDDisplayInteractor()
{
}


//-----------------------------------------------------------------------------
void mitk::DnDDisplayInteractor::Notify(InteractionEvent* interactionEvent, bool isHandled)
{
  mitk::BaseRenderer* renderer = interactionEvent->GetSender();
  if (std::find(m_Renderers.begin(), m_Renderers.end(), renderer) != m_Renderers.end())
  {
    Superclass::Notify(interactionEvent, isHandled);
  }
}


//-----------------------------------------------------------------------------
void mitk::DnDDisplayInteractor::ConnectActionsAndFunctions()
{
  Superclass::ConnectActionsAndFunctions();
  CONNECT_FUNCTION("selectPosition", SelectPosition);
  CONNECT_FUNCTION("initMove", InitMove);
  CONNECT_FUNCTION("initZoom", InitZoom);
  CONNECT_FUNCTION("setWindowLayoutToAxial", SetWindowLayoutToAxial);
  CONNECT_FUNCTION("setWindowLayoutToSagittal", SetWindowLayoutToSagittal);
  CONNECT_FUNCTION("setWindowLayoutToCoronal", SetWindowLayoutToCoronal);
  CONNECT_FUNCTION("setWindowLayoutTo3D", SetWindowLayoutTo3D);
  CONNECT_FUNCTION("setWindowLayoutToMulti", SetWindowLayoutToMulti);
  CONNECT_FUNCTION("toggleMultiWindowLayout", ToggleMultiWindowLayout);
  CONNECT_FUNCTION("toggleCursorVisibility", ToggleCursorVisibility);
  CONNECT_FUNCTION("selectPreviousTimeStep", SelectPreviousTimeStep);
  CONNECT_FUNCTION("selectNextTimeStep", SelectNextTimeStep);
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* mitk::DnDDisplayInteractor::GetRenderWindow(mitk::BaseRenderer* renderer)
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
bool mitk::DnDDisplayInteractor::SelectPosition(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  mitk::BaseRenderer* renderer = interactionEvent->GetSender();

  /// Note:
  /// We do not re-implement position selection for the 3D window.
  if (renderer == m_Renderers[3])
  {
    return false;
  }

  InteractionPositionEvent* positionEvent = dynamic_cast<InteractionPositionEvent*>(interactionEvent);
  if (positionEvent == NULL)
  {
    MITK_WARN << "mitk DnDDisplayInteractor cannot process the event: " << interactionEvent->GetNameOfClass();
    return false;
  }

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
  // In the niftkMultiWindowWidget this puts the crosshair to the mouse position, and
  // selects the slice in the two other render window.
  const mitk::Point3D& positionInWorld = positionEvent->GetPositionInWorld();
  m_Viewer->SetSelectedPosition(positionInWorld);

  m_Viewer->BlockUpdate(updateWasBlocked);

  return true;
}


//-----------------------------------------------------------------------------
bool mitk::DnDDisplayInteractor::ScrollOneUp(StateMachineAction* action, InteractionEvent* interactionEvent)
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
  ///   niftkSingleViewerWidget::SetNavigationControllerEventListening(bool)
  /// and
  ///   QmitkMultiWindowWidget::SetWidgetPlanesLocked(bool)

//  bool result = Superclass::ScrollOneUp(action, interactionEvent);

  m_Viewer->MoveAnterior();

  m_Viewer->BlockUpdate(updateWasBlocked);

//  return result;
  return true;
}


//-----------------------------------------------------------------------------
bool mitk::DnDDisplayInteractor::ScrollOneDown(StateMachineAction* action, InteractionEvent* interactionEvent)
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
  ///   niftkSingleViewerWidget::SetNavigationControllerEventListening(bool)
  /// and
  ///   QmitkMultiWindowWidget::SetWidgetPlanesLocked(bool)

//  bool result = Superclass::ScrollOneDown(action, interactionEvent);

  m_Viewer->MovePosterior();

  m_Viewer->BlockUpdate(updateWasBlocked);

//  return result;
  return true;
}


//-----------------------------------------------------------------------------
bool mitk::DnDDisplayInteractor::InitMove(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  InteractionPositionEvent* positionEvent = dynamic_cast<InteractionPositionEvent*>(interactionEvent);
  if (positionEvent == NULL)
  {
    MITK_WARN << "mitk DnDDisplayInteractor cannot process the event: " << interactionEvent->GetNameOfClass();
    return false;
  }

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


//-----------------------------------------------------------------------------
bool mitk::DnDDisplayInteractor::InitZoom(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  InteractionPositionEvent* positionEvent = dynamic_cast<InteractionPositionEvent*>(interactionEvent);
  if (positionEvent == NULL)
  {
    MITK_WARN << "mitk DnDDisplayInteractor cannot process the event: " << interactionEvent->GetNameOfClass();
    return false;
  }

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


//-----------------------------------------------------------------------------
bool mitk::DnDDisplayInteractor::SetWindowLayoutToAxial(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  m_Viewer->SetWindowLayout(WINDOW_LAYOUT_AXIAL);
  return true;
}


//-----------------------------------------------------------------------------
bool mitk::DnDDisplayInteractor::SetWindowLayoutToSagittal(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  m_Viewer->SetWindowLayout(WINDOW_LAYOUT_SAGITTAL);
  return true;
}


//-----------------------------------------------------------------------------
bool mitk::DnDDisplayInteractor::SetWindowLayoutToCoronal(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  m_Viewer->SetWindowLayout(WINDOW_LAYOUT_CORONAL);
  return true;
}


//-----------------------------------------------------------------------------
bool mitk::DnDDisplayInteractor::SetWindowLayoutTo3D(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  m_Viewer->SetWindowLayout(WINDOW_LAYOUT_3D);
  return true;
}


//-----------------------------------------------------------------------------
bool mitk::DnDDisplayInteractor::SetWindowLayoutToMulti(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  m_Viewer->ToggleMultiWindowLayout();
  return true;
}


//-----------------------------------------------------------------------------
bool mitk::DnDDisplayInteractor::ToggleMultiWindowLayout(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  m_Viewer->ToggleMultiWindowLayout();
  return true;
}


//-----------------------------------------------------------------------------
bool mitk::DnDDisplayInteractor::ToggleCursorVisibility(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  m_Viewer->ToggleCursorVisibility();
  return true;
}


//-----------------------------------------------------------------------------
bool mitk::DnDDisplayInteractor::SelectPreviousTimeStep(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  int timeStep = m_Viewer->GetTimeStep() - 1;

  if (timeStep >= 0)
  {
    m_Viewer->SetTimeStep(timeStep);
  }

  return timeStep == m_Viewer->GetTimeStep();
}


//-----------------------------------------------------------------------------
bool mitk::DnDDisplayInteractor::SelectNextTimeStep(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  int timeStep = m_Viewer->GetTimeStep() + 1;

  if (timeStep <= m_Viewer->GetMaxTimeStep())
  {
    m_Viewer->SetTimeStep(timeStep);
  }

  return timeStep == m_Viewer->GetTimeStep();
}

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
#include <mitkInteractionPositionEvent.h>
#include <mitkLine.h>
#include <mitkSliceNavigationController.h>
#include <mitkGlobalInteraction.h>


//-----------------------------------------------------------------------------
mitk::DnDDisplayInteractor::DnDDisplayInteractor(niftkSingleViewerWidget* multiWindowWidget)
: mitk::DisplayInteractor()
, m_MultiWindowWidget(multiWindowWidget)
, m_Renderers(3)
, m_FocusManager(mitk::GlobalInteraction::GetInstance()->GetFocusManager())
{
  const std::vector<QmitkRenderWindow*>& renderWindows = m_MultiWindowWidget->GetRenderWindows();
  m_Renderers[0] = renderWindows[0]->GetRenderer();
  m_Renderers[1] = renderWindows[1]->GetRenderer();
  m_Renderers[2] = renderWindows[2]->GetRenderer();
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
  mitk::DisplayInteractor::ConnectActionsAndFunctions();
  CONNECT_FUNCTION("selectPosition", SelectPosition);
  CONNECT_FUNCTION("initMove", InitMove);
  CONNECT_FUNCTION("initZoom", InitZoom);
}


//-----------------------------------------------------------------------------
QmitkRenderWindow* mitk::DnDDisplayInteractor::GetRenderWindow(mitk::BaseRenderer* renderer)
{
  QmitkRenderWindow* renderWindow = 0;

  std::size_t i = std::find(m_Renderers.begin(), m_Renderers.end(), renderer) - m_Renderers.begin();

  if (i < 3)
  {
    renderWindow = m_MultiWindowWidget->GetRenderWindows()[i];
  }

  return renderWindow;
}


//-----------------------------------------------------------------------------
int mitk::DnDDisplayInteractor::GetOrientation(mitk::BaseRenderer* renderer)
{
  return std::find(m_Renderers.begin(), m_Renderers.end(), renderer) - m_Renderers.begin();
}


//-----------------------------------------------------------------------------
bool mitk::DnDDisplayInteractor::SelectPosition(StateMachineAction* /*action*/, InteractionEvent* interactionEvent)
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

  bool updateWasBlocked = m_MultiWindowWidget->BlockUpdate(true);

  mitk::BaseRenderer* renderer = interactionEvent->GetSender();
  if (renderer != m_FocusManager->GetFocused())
  {
    QmitkRenderWindow* renderWindow = this->GetRenderWindow(renderer);
    m_MultiWindowWidget->SetSelectedRenderWindow(renderWindow);
    m_MultiWindowWidget->SetFocused();
  }

  // Selects the point under the mouse pointer in the slice navigation controllers.
  // In the niftkMultiWindowWidget this puts the crosshair to the mouse position, and
  // selects the slice in the two other render window.
  const mitk::Point3D& positionInWorld = positionEvent->GetPositionInWorld();
  m_MultiWindowWidget->SetSelectedPosition(positionInWorld);

  m_MultiWindowWidget->BlockUpdate(updateWasBlocked);

  return true;
}


//-----------------------------------------------------------------------------
bool mitk::DnDDisplayInteractor::ScrollOneUp(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  bool updateWasBlocked = m_MultiWindowWidget->BlockUpdate(true);

  mitk::BaseRenderer* renderer = interactionEvent->GetSender();
  if (renderer != m_FocusManager->GetFocused())
  {
    QmitkRenderWindow* renderWindow = this->GetRenderWindow(renderer);
    m_MultiWindowWidget->SetSelectedRenderWindow(renderWindow);
    m_MultiWindowWidget->SetFocused();
  }

  /// Note:
  /// This does not work if the slice are locked.
  /// See:
  ///   niftkSingleViewerWidget::SetNavigationControllerEventListening(bool)
  /// and
  ///   QmitkMultiWindowWidget::SetWidgetPlanesLocked(bool)

//  bool result = Superclass::ScrollOneUp(action, interactionEvent);

  m_MultiWindowWidget->MoveAnterior();

  m_MultiWindowWidget->BlockUpdate(updateWasBlocked);

//  return result;
  return true;
}


//-----------------------------------------------------------------------------
bool mitk::DnDDisplayInteractor::ScrollOneDown(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  bool updateWasBlocked = m_MultiWindowWidget->BlockUpdate(true);

  mitk::BaseRenderer* renderer = interactionEvent->GetSender();
  if (renderer != m_FocusManager->GetFocused())
  {
    QmitkRenderWindow* renderWindow = this->GetRenderWindow(renderer);
    m_MultiWindowWidget->SetSelectedRenderWindow(renderWindow);
    m_MultiWindowWidget->SetFocused();
  }

  /// Note:
  /// This does not work if the slice are locked.
  /// See:
  ///   niftkSingleViewerWidget::SetNavigationControllerEventListening(bool)
  /// and
  ///   QmitkMultiWindowWidget::SetWidgetPlanesLocked(bool)

//  bool result = Superclass::ScrollOneDown(action, interactionEvent);

  m_MultiWindowWidget->MovePosterior();

  m_MultiWindowWidget->BlockUpdate(updateWasBlocked);

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

  bool updateWasBlocked = m_MultiWindowWidget->BlockUpdate(true);

  mitk::BaseRenderer* renderer = interactionEvent->GetSender();
  if (renderer != m_FocusManager->GetFocused())
  {
    QmitkRenderWindow* renderWindow = this->GetRenderWindow(renderer);
    m_MultiWindowWidget->SetSelectedRenderWindow(renderWindow);
    m_MultiWindowWidget->SetFocused();
  }

  bool result = this->Init(action, interactionEvent);

  m_MultiWindowWidget->BlockUpdate(updateWasBlocked);

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

  bool updateWasBlocked = m_MultiWindowWidget->BlockUpdate(true);

  mitk::BaseRenderer* renderer = interactionEvent->GetSender();
  if (renderer != m_FocusManager->GetFocused())
  {
    QmitkRenderWindow* renderWindow = this->GetRenderWindow(renderer);
    m_MultiWindowWidget->SetSelectedRenderWindow(renderWindow);
    m_MultiWindowWidget->SetFocused();
  }

  /// Note that the zoom focus must always be the selected position,
  /// i.e. the position at the cursor (crosshair).
  mitk::Point3D focusPoint3DInMm = m_MultiWindowWidget->GetSelectedPosition();

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

  m_MultiWindowWidget->BlockUpdate(updateWasBlocked);

  return result;
}

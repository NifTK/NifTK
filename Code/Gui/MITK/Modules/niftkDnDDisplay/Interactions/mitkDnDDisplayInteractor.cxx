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

#include "../niftkMultiWindowWidget_p.h"

#include <mitkBaseRenderer.h>
#include <mitkInteractionPositionEvent.h>
#include <mitkLine.h>
#include <mitkSliceNavigationController.h>
#include <mitkGlobalInteraction.h>

mitk::DnDDisplayInteractor::DnDDisplayInteractor(niftkMultiWindowWidget* multiWindowWidget)
: mitk::DisplayInteractor()
, m_MultiWindowWidget(multiWindowWidget)
, m_Renderers(4)
{
  m_Renderers[0] = m_MultiWindowWidget->GetRenderWindow1()->GetRenderer();
  m_Renderers[1] = m_MultiWindowWidget->GetRenderWindow2()->GetRenderer();
  m_Renderers[2] = m_MultiWindowWidget->GetRenderWindow3()->GetRenderer();
  m_Renderers[3] = m_MultiWindowWidget->GetRenderWindow4()->GetRenderer();
}

mitk::DnDDisplayInteractor::~DnDDisplayInteractor()
{
}

void mitk::DnDDisplayInteractor::Notify(InteractionEvent* interactionEvent, bool isHandled)
{
  mitk::BaseRenderer* renderer = interactionEvent->GetSender();
  if (std::find(m_Renderers.begin(), m_Renderers.end(), renderer) != m_Renderers.end())
  {
    Superclass::Notify(interactionEvent, isHandled);
  }
}

void mitk::DnDDisplayInteractor::ConnectActionsAndFunctions()
{
  mitk::DisplayInteractor::ConnectActionsAndFunctions();
  CONNECT_FUNCTION("selectPosition", SelectPosition);
  CONNECT_FUNCTION("initZoom", InitZoom);
}


QmitkRenderWindow* mitk::DnDDisplayInteractor::GetRenderWindow(mitk::BaseRenderer* renderer)
{
  QmitkRenderWindow* renderWindow = 0;
  if (renderer == m_Renderers[0])
  {
    renderWindow = m_MultiWindowWidget->GetRenderWindow1();
  }
  else if (renderer == m_Renderers[1])
  {
    renderWindow = m_MultiWindowWidget->GetRenderWindow2();
  }
  else if (renderer == m_Renderers[2])
  {
    renderWindow = m_MultiWindowWidget->GetRenderWindow3();
  }
  else if (renderer == m_Renderers[3])
  {
    renderWindow = m_MultiWindowWidget->GetRenderWindow4();
  }
  return renderWindow;
}

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
  if (!renderer->GetFocused())
  {
    QmitkRenderWindow* renderWindow = this->GetRenderWindow(renderer);
    m_MultiWindowWidget->SetSelectedRenderWindow(renderWindow);
  }

  // Selects the point under the mouse pointer in the slice navigation controllers.
  // In the niftkMultiWindowWidget this puts the crosshair to the mouse position, and
  // selects the slice in the two other render window.
  const mitk::Point3D& positionInWorld = positionEvent->GetPositionInWorld();
  m_MultiWindowWidget->SetSelectedPosition(positionInWorld);

  m_MultiWindowWidget->BlockUpdate(updateWasBlocked);

  return true;
}


bool mitk::DnDDisplayInteractor::ScrollOneUp(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  bool updateWasBlocked = m_MultiWindowWidget->BlockUpdate(true);

  mitk::BaseRenderer* renderer = interactionEvent->GetSender();
  if (!renderer->GetFocused())
  {
    QmitkRenderWindow* renderWindow = this->GetRenderWindow(renderer);
    m_MultiWindowWidget->SetSelectedRenderWindow(renderWindow);
  }

  bool result = Superclass::ScrollOneUp(action, interactionEvent);

  m_MultiWindowWidget->BlockUpdate(updateWasBlocked);

  return result;
}


bool mitk::DnDDisplayInteractor::ScrollOneDown(StateMachineAction* action, InteractionEvent* interactionEvent)
{
  bool updateWasBlocked = m_MultiWindowWidget->BlockUpdate(true);

  mitk::BaseRenderer* renderer = interactionEvent->GetSender();
  if (!renderer->GetFocused())
  {
    QmitkRenderWindow* renderWindow = this->GetRenderWindow(renderer);
    m_MultiWindowWidget->SetSelectedRenderWindow(renderWindow);
  }

  bool result = Superclass::ScrollOneDown(action, interactionEvent);

  m_MultiWindowWidget->BlockUpdate(updateWasBlocked);

  return result;
}


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
  if (!renderer->GetFocused())
  {
    QmitkRenderWindow* renderWindow = this->GetRenderWindow(renderer);
    m_MultiWindowWidget->SetSelectedRenderWindow(renderWindow);
  }

  // Selects the point under the mouse pointer in the slice navigation controllers.
  // In the niftkMultiWindowWidget this puts the crosshair to the mouse position, and
  // selects the slice in the two other render window.
  const mitk::Point3D& positionInWorld = positionEvent->GetPositionInWorld();
  m_MultiWindowWidget->SetSelectedPosition(positionInWorld);

  // Although the code above puts the crosshair to the mouse pointer position,
  // the two positions are not completely equal because the crosshair is always in
  // the middle of the voxel that contains the mouse position. This slight difference
  // causes that in strong zooming the crosshair moves away from the focus point.
  // So that we zoom around the crosshair, we have to calculate the crosshair position
  // (in world coordinates) and then its projection to the displayed region (in pixels).
  // This will be the focus point during the zooming.
  const mitk::PlaneGeometry* plane1 = m_Renderers[0]->GetSliceNavigationController()->GetCurrentPlaneGeometry();
  const mitk::PlaneGeometry* plane2 = m_Renderers[1]->GetSliceNavigationController()->GetCurrentPlaneGeometry();
  const mitk::PlaneGeometry* plane3 = m_Renderers[2]->GetSliceNavigationController()->GetCurrentPlaneGeometry();

  mitk::Line3D intersectionLine;
  mitk::Point3D focusPoint3DInMm;
  if (!(plane1 && plane2 && plane1->IntersectionLine(plane2, intersectionLine) &&
        plane3 && plane3->IntersectionPoint(intersectionLine, focusPoint3DInMm)))
  {
    focusPoint3DInMm = positionInWorld;
  }

  mitk::Point2D focusPoint2DInMm;
  mitk::Point2D focusPoint2DInPx;
  mitk::Point2D focusPoint2DInPxUL;

  mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();
  displayGeometry->Map(focusPoint3DInMm, focusPoint2DInMm);
  displayGeometry->WorldToDisplay(focusPoint2DInMm, focusPoint2DInPx);
  displayGeometry->DisplayToULDisplay(focusPoint2DInPx, focusPoint2DInPxUL);

  // Create a new position event with the "corrected" position.
  mitk::InteractionPositionEvent::Pointer positionEvent2 = InteractionPositionEvent::New(renderer, focusPoint2DInPxUL);

  bool result = this->Init(action, positionEvent2);

  m_MultiWindowWidget->BlockUpdate(updateWasBlocked);

  return result;
}

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASDisplayInteractor.h"

#include <string.h>

#include <mitkBaseRenderer.h>
#include <mitkInteractionPositionEvent.h>
#include <mitkLevelWindow.h>
#include <mitkLevelWindowProperty.h>
#include <mitkLine.h>
#include <mitkNodePredicateDataType.h>
#include <mitkPropertyList.h>
#include <mitkSliceNavigationController.h>
#include <mitkStandaloneDataStorage.h>

mitk::MIDASDisplayInteractor::MIDASDisplayInteractor(const std::vector<mitk::BaseRenderer*>& renderers, const std::vector<mitk::SliceNavigationController*>& sliceNavigationControllers)
: m_IndexToSliceModifier(4)
, m_AutoRepeat(false)
, m_AlwaysReact(false)
, m_ZoomFactor(2)
, m_Renderers(renderers)
, m_SliceNavigationControllers(sliceNavigationControllers)
{
  // MIDAS customisation:
  // This interactor works with the MIDASStdMultiWidget, but it is decoupled from it.
  // (No GUI dependence.) The slice navigation controllers should be the axial, sagittal
  // and coronal SNCs of the MIDASStdMultiWidget.
  assert(sliceNavigationControllers.size() == 3);

  m_StartDisplayCoordinate.Fill(0);
  m_LastDisplayCoordinate.Fill(0);
  m_CurrentDisplayCoordinate.Fill(0);
}

mitk::MIDASDisplayInteractor::~MIDASDisplayInteractor()
{
}

void mitk::MIDASDisplayInteractor::Notify(InteractionEvent* interactionEvent, bool isHandled)
{
  // to use the state machine pattern,
  // the event is passed to the state machine interface to be handled
  if (!isHandled || m_AlwaysReact)
  {
    mitk::BaseRenderer* sender = interactionEvent->GetSender();
    if (std::find(m_Renderers.begin(), m_Renderers.end(), sender) != m_Renderers.end())
    {
      this->HandleEvent(interactionEvent, NULL);
    }
  }
}

void mitk::MIDASDisplayInteractor::ConnectActionsAndFunctions()
{
  CONNECT_FUNCTION("init", Init);
  CONNECT_FUNCTION("initZoom", InitZoom);
  CONNECT_FUNCTION("move", Move);
  CONNECT_FUNCTION("zoom", Zoom);
  CONNECT_FUNCTION("scroll", Scroll);
  CONNECT_FUNCTION("ScrollOneDown", ScrollOneDown);
  CONNECT_FUNCTION("ScrollOneUp", ScrollOneUp);
  CONNECT_FUNCTION("levelWindow", AdjustLevelWindow);
}

bool mitk::MIDASDisplayInteractor::Init(StateMachineAction*, InteractionEvent* interactionEvent)
{
  MITK_INFO << "mitk::MIDASDisplayInteractor::Init(StateMachineAction*, InteractionEvent* interactionEvent)" << std::endl;
  BaseRenderer* sender = interactionEvent->GetSender();
  InteractionPositionEvent* positionEvent = dynamic_cast<InteractionPositionEvent*>(interactionEvent);
  if (positionEvent == NULL)
  {
    MITK_WARN<< "DisplayInteractor cannot process the event: " << interactionEvent->GetNameOfClass();
    return false;
  }

  // --------------------------------------------------------------------------
  // MIDAS customisation starts.
  //
  // First, check if the slice navigation controllers have a valid geometry,
  // i.e. an image is loaded.
//  if (!m_SliceNavigationControllers[0]->GetCreatedWorldGeometry())
//  {
//    return false;
//  }

//  // Selects the point under the mouse pointer in the slice navigation controllers.
//  // In the MIDASStdMultiWidget this puts the crosshair to the mouse position, and
//  // selects the slice in the two other render window.
//  const mitk::Point3D& positionInWorld = positionEvent->GetPositionInWorld();
//  m_SliceNavigationControllers[0]->SelectSliceByPoint(positionInWorld);
//  m_SliceNavigationControllers[1]->SelectSliceByPoint(positionInWorld);
//  m_SliceNavigationControllers[2]->SelectSliceByPoint(positionInWorld);

//  // Although the code above puts the crosshair to the mouse pointer position,
//  // the two positions are not completely equal because the crosshair is always in
//  // the middle of the voxel that contains the mouse position. This slight difference
//  // causes that in strong zooming the crosshair moves away from the focus point.
//  // So that we zoom around the crosshair, we have to calculate the crosshair position
//  // (in world coordinates) and then its projection to the displayed region (in pixels).
//  // This will be the focus point during the zooming.
//  const mitk::PlaneGeometry* plane1 = m_SliceNavigationControllers[0]->GetCurrentPlaneGeometry();
//  const mitk::PlaneGeometry* plane2 = m_SliceNavigationControllers[1]->GetCurrentPlaneGeometry();
//  const mitk::PlaneGeometry* plane3 = m_SliceNavigationControllers[2]->GetCurrentPlaneGeometry();

//  mitk::Line3D line;
//  mitk::Point3D point;
//  mitk::Point3D focusPoint;
//  if (plane1 && plane2 && plane1->IntersectionLine(plane2, line) &&
//      plane3 && plane3->IntersectionPoint(line, point))
//  {
//    focusPoint = point;
//  }
//  else
//  {
//    focusPoint = positionInWorld;
//  }

//  mitk::Point2D projectedFocusInMillimeters;
//  mitk::Point2D projectedFocusInPixels;

//  mitk::DisplayGeometry* displayGeometry = sender->GetDisplayGeometry();
//  displayGeometry->Map(focusPoint, projectedFocusInMillimeters);
//  displayGeometry->WorldToDisplay(projectedFocusInMillimeters, projectedFocusInPixels);

//  m_StartDisplayCoordinate = projectedFocusInPixels;
//  m_LastDisplayCoordinate = projectedFocusInPixels;
//  m_CurrentDisplayCoordinate = projectedFocusInPixels;

  //
  // MIDAS customisation ends.
  // --------------------------------------------------------------------------

  // Original MITK code:
  mitk::DisplayGeometry* displayGeometry = sender->GetDisplayGeometry();
  m_StartDisplayCoordinate = positionEvent->GetPointerPositionOnScreen();
  m_LastDisplayCoordinate = positionEvent->GetPointerPositionOnScreen();
  m_CurrentDisplayCoordinate = positionEvent->GetPointerPositionOnScreen();

  Vector2D origin = displayGeometry->GetOriginInMM();
  double scaleFactorMMPerDisplayUnit = displayGeometry->GetScaleFactorMMPerDisplayUnit();

  m_StartCoordinateInMM = mitk::Point2D(
      (origin + m_StartDisplayCoordinate.GetVectorFromOrigin() * scaleFactorMMPerDisplayUnit).GetDataPointer());

  return true;
}

bool mitk::MIDASDisplayInteractor::InitZoom(StateMachineAction*, InteractionEvent* interactionEvent)
{
  MITK_INFO << "mitk::MIDASDisplayInteractor::InitZoom(StateMachineAction*, InteractionEvent* interactionEvent)" << std::endl;
  BaseRenderer* sender = interactionEvent->GetSender();
  InteractionPositionEvent* positionEvent = dynamic_cast<InteractionPositionEvent*>(interactionEvent);
  if (positionEvent == NULL)
  {
    MITK_WARN<< "DisplayInteractor cannot process the event: " << interactionEvent->GetNameOfClass();
    return false;
  }

  // --------------------------------------------------------------------------
  // MIDAS customisation starts.
  //
  // First, check if the slice navigation controllers have a valid geometry,
  // i.e. an image is loaded.
  if (!m_SliceNavigationControllers[0]->GetCreatedWorldGeometry())
  {
    return false;
  }

  // Selects the point under the mouse pointer in the slice navigation controllers.
  // In the MIDASStdMultiWidget this puts the crosshair to the mouse position, and
  // selects the slice in the two other render window.
  const mitk::Point3D& positionInWorld = positionEvent->GetPositionInWorld();
  m_SliceNavigationControllers[0]->SelectSliceByPoint(positionInWorld);
  m_SliceNavigationControllers[1]->SelectSliceByPoint(positionInWorld);
  m_SliceNavigationControllers[2]->SelectSliceByPoint(positionInWorld);

  // Although the code above puts the crosshair to the mouse pointer position,
  // the two positions are not completely equal because the crosshair is always in
  // the middle of the voxel that contains the mouse position. This slight difference
  // causes that in strong zooming the crosshair moves away from the focus point.
  // So that we zoom around the crosshair, we have to calculate the crosshair position
  // (in world coordinates) and then its projection to the displayed region (in pixels).
  // This will be the focus point during the zooming.
  const mitk::PlaneGeometry* plane1 = m_SliceNavigationControllers[0]->GetCurrentPlaneGeometry();
  const mitk::PlaneGeometry* plane2 = m_SliceNavigationControllers[1]->GetCurrentPlaneGeometry();
  const mitk::PlaneGeometry* plane3 = m_SliceNavigationControllers[2]->GetCurrentPlaneGeometry();

  mitk::Line3D line;
  mitk::Point3D point;
  mitk::Point3D focusPoint;
  if (plane1 && plane2 && plane1->IntersectionLine(plane2, line) &&
      plane3 && plane3->IntersectionPoint(line, point))
  {
    focusPoint = point;
  }
  else
  {
    focusPoint = positionInWorld;
  }

  mitk::Point2D projectedFocusInMillimeters;
  mitk::Point2D projectedFocusInPixels;

  mitk::DisplayGeometry* displayGeometry = sender->GetDisplayGeometry();
  displayGeometry->Map(focusPoint, projectedFocusInMillimeters);
  displayGeometry->WorldToDisplay(projectedFocusInMillimeters, projectedFocusInPixels);

  m_StartDisplayCoordinate = projectedFocusInPixels;
  m_LastDisplayCoordinate = projectedFocusInPixels;
  m_CurrentDisplayCoordinate = projectedFocusInPixels;

  //
  // MIDAS customisation ends.
  // --------------------------------------------------------------------------

  // Original MITK code:
//  m_StartDisplayCoordinate = positionEvent->GetPointerPositionOnScreen();
//  m_LastDisplayCoordinate = positionEvent->GetPointerPositionOnScreen();
//  m_CurrentDisplayCoordinate = positionEvent->GetPointerPositionOnScreen();

  Vector2D origin = displayGeometry->GetOriginInMM();
  double scaleFactorMMPerDisplayUnit = displayGeometry->GetScaleFactorMMPerDisplayUnit();

  m_StartCoordinateInMM = mitk::Point2D(
      (origin + m_StartDisplayCoordinate.GetVectorFromOrigin() * scaleFactorMMPerDisplayUnit).GetDataPointer());

  return true;
}

bool mitk::MIDASDisplayInteractor::Move(StateMachineAction*, InteractionEvent* interactionEvent)
{
  BaseRenderer* sender = interactionEvent->GetSender();
  InteractionPositionEvent* positionEvent = dynamic_cast<InteractionPositionEvent*>(interactionEvent);
  if (positionEvent == NULL)
  {
    MITK_WARN<< "DisplayInteractor: cannot process the event in Move action: " << interactionEvent->GetNameOfClass();
    return false;
  }
  // perform translation
  sender->GetDisplayGeometry()->MoveBy((positionEvent->GetPointerPositionOnScreen() - m_LastDisplayCoordinate) * (-1.0));
  sender->GetRenderingManager()->RequestUpdate(sender->GetRenderWindow());
  m_LastDisplayCoordinate = positionEvent->GetPointerPositionOnScreen();
  return true;
}

bool mitk::MIDASDisplayInteractor::Zoom(StateMachineAction*, InteractionEvent* interactionEvent)
{
  const BaseRenderer::Pointer sender = interactionEvent->GetSender();
  InteractionPositionEvent* positionEvent = dynamic_cast<InteractionPositionEvent*>(interactionEvent);
  if (positionEvent == NULL)
  {
    MITK_WARN<< "DisplayInteractor cannot process the event: " << interactionEvent->GetNameOfClass();
    return false;
  }
  float factor = 1.0;
  float distance = 0;
  if (m_ZoomDirection == "leftright")
  {
    distance = m_CurrentDisplayCoordinate[1] - m_LastDisplayCoordinate[1];
  }
  else
  {
    distance = m_CurrentDisplayCoordinate[0] - m_LastDisplayCoordinate[0];
  }
  // set zooming speed
  if (distance < 0.0)
  {
    factor = 1.0 / m_ZoomFactor;
  }
  else if (distance > 0.0)
  {
    factor = 1.0 * m_ZoomFactor;
  }
  sender->GetDisplayGeometry()->ZoomWithFixedWorldCoordinates(factor, m_StartDisplayCoordinate, m_StartCoordinateInMM);
  sender->GetRenderingManager()->RequestUpdate(sender->GetRenderWindow());
  m_LastDisplayCoordinate = m_CurrentDisplayCoordinate;
  m_CurrentDisplayCoordinate = positionEvent->GetPointerPositionOnScreen();
  return true;
}

bool mitk::MIDASDisplayInteractor::Scroll(StateMachineAction*, InteractionEvent* interactionEvent)
{
  InteractionPositionEvent* positionEvent = dynamic_cast<InteractionPositionEvent*>(interactionEvent);
  if (positionEvent == NULL)
  {
    MITK_WARN<< "DisplayInteractor::Scroll cannot process the event: " << interactionEvent->GetNameOfClass();
    return false;
  }

  mitk::SliceNavigationController::Pointer sliceNaviController = interactionEvent->GetSender()->GetSliceNavigationController();
  if (sliceNaviController)
  {
    int delta = 0;
    // Scrolling direction
    if (m_ScrollDirection == "leftright")
    {
      delta = static_cast<int>(m_LastDisplayCoordinate[1] - positionEvent->GetPointerPositionOnScreen()[1]);
    }
    else
    {
      delta = static_cast<int>(m_LastDisplayCoordinate[0] - positionEvent->GetPointerPositionOnScreen()[0]);
    }
    // Set how many pixels the mouse has to be moved to scroll one slice
    // if we moved less than 'm_IndexToSliceModifier' pixels slice ONE slice only
    if (delta > 0 && delta < m_IndexToSliceModifier)
    {
      delta = m_IndexToSliceModifier;
    }
    else if (delta < 0 && delta > -m_IndexToSliceModifier)
    {
      delta = -m_IndexToSliceModifier;
    }
    delta /= m_IndexToSliceModifier;

    int newPos = sliceNaviController->GetSlice()->GetPos() + delta;

    // if auto repeat is on, start at first slice if you reach the last slice and vice versa
    int maxSlices = sliceNaviController->GetSlice()->GetSteps();
    if (m_AutoRepeat)
    {
      while (newPos < 0)
      {
        newPos += maxSlices;
      }

      while (newPos >= maxSlices)
      {
        newPos -= maxSlices;
      }
    }
    else
    {
      // if the new slice is below 0 we still show slice 0
      // due to the stepper using unsigned int we have to do this ourselves
      if (newPos < 1)
      {
        newPos = 0;
      }
    }
    // set the new position
    sliceNaviController->GetSlice()->SetPos(newPos);
    m_LastDisplayCoordinate = m_CurrentDisplayCoordinate;
    m_CurrentDisplayCoordinate = positionEvent->GetPointerPositionOnScreen();
  }
  return true;
}

bool mitk::MIDASDisplayInteractor::ScrollOneDown(StateMachineAction*, InteractionEvent* interactionEvent)
{
  mitk::SliceNavigationController::Pointer sliceNaviController = interactionEvent->GetSender()->GetSliceNavigationController();
  if (!sliceNaviController->GetSliceLocked())
  {
    mitk::Stepper* stepper = sliceNaviController->GetSlice();
    if (stepper->GetSteps() <= 1)
    {
      stepper = sliceNaviController->GetTime();
    }
    stepper->Next();
  }
  return true;
}

bool mitk::MIDASDisplayInteractor::ScrollOneUp(StateMachineAction*, InteractionEvent* interactionEvent)
{
  mitk::SliceNavigationController::Pointer sliceNaviController = interactionEvent->GetSender()->GetSliceNavigationController();
  if (!sliceNaviController->GetSliceLocked())
  {
    mitk::Stepper* stepper = sliceNaviController->GetSlice();
    if (stepper->GetSteps() <= 1)
    {
      stepper = sliceNaviController->GetTime();
    }
    stepper->Previous();
    return true;
  }
  return false;
}

bool mitk::MIDASDisplayInteractor::AdjustLevelWindow(StateMachineAction*, InteractionEvent* interactionEvent)
{
  BaseRenderer::Pointer sender = interactionEvent->GetSender();
  InteractionPositionEvent* positionEvent = dynamic_cast<InteractionPositionEvent*>(interactionEvent);
  if (positionEvent == NULL)
  {
    MITK_WARN<< "DisplayInteractor::Scroll cannot process the event: " << interactionEvent->GetNameOfClass();
    return false;
  }
  m_LastDisplayCoordinate = m_CurrentDisplayCoordinate;
  m_CurrentDisplayCoordinate = positionEvent->GetPointerPositionOnScreen();
  // search for active image
  mitk::DataStorage::Pointer storage = sender->GetDataStorage();
  mitk::DataNode::Pointer node = NULL;
  mitk::DataStorage::SetOfObjects::ConstPointer allImageNodes = storage->GetSubset(mitk::NodePredicateDataType::New("Image"));
  for (unsigned int i = 0; i < allImageNodes->size(); i++)
  {
    bool isActiveImage = false;
    bool propFound = allImageNodes->at(i)->GetBoolProperty("imageForLevelWindow", isActiveImage);

    if (propFound && isActiveImage)
    {
      node = allImageNodes->at(i);
      continue;
    }
  }
  if (node.IsNull())
  {
    node = storage->GetNode(mitk::NodePredicateDataType::New("Image"));
  }
  if (node.IsNull())
  {
    return false;
  }

  mitk::LevelWindow lv = mitk::LevelWindow();
  node->GetLevelWindow(lv);
  ScalarType level = lv.GetLevel();
  ScalarType window = lv.GetWindow();
  // calculate adjustments from mouse movements
  level += (m_CurrentDisplayCoordinate[0] - m_LastDisplayCoordinate[0]) * static_cast<ScalarType>(2);
  window += (m_CurrentDisplayCoordinate[1] - m_LastDisplayCoordinate[1]) * static_cast<ScalarType>(2);

  lv.SetLevelWindow(level, window);
  dynamic_cast<mitk::LevelWindowProperty*>(node->GetProperty("levelwindow"))->SetLevelWindow(lv);

  sender->GetRenderingManager()->RequestUpdateAll();
  return true;
}

void mitk::MIDASDisplayInteractor::ConfigurationChanged()
{
  mitk::PropertyList::Pointer properties = GetAttributes();
  // auto repeat
  std::string strAutoRepeat = "";
  if (properties->GetStringProperty("autoRepeat", strAutoRepeat))
  {
    if (strAutoRepeat == "true")
    {
      m_AutoRepeat = true;
    }
    else
    {
      m_AutoRepeat = false;
    }
  }
  // pixel movement for scrolling one slice
  std::string strPixelPerSlice = "";
  if (properties->GetStringProperty("pixelPerSlice", strPixelPerSlice))
  {
    m_IndexToSliceModifier = atoi(strPixelPerSlice.c_str());
  }
  else
  {
    m_IndexToSliceModifier = 4;
  }
  // scroll direction
  if (!properties->GetStringProperty("zoomDirection", m_ScrollDirection))
  {
    m_ScrollDirection = "updown";
  }
  // zoom direction
  if (!properties->GetStringProperty("zoomDirection", m_ZoomDirection))
  {
    m_ZoomDirection = "updown";
  }
  // zoom factor
  std::string strZoomFactor = "";
  properties->GetStringProperty("zoomFactor", strZoomFactor);
  m_ZoomFactor = .05;
  if (atoi(strZoomFactor.c_str()) > 0)
  {
    m_ZoomFactor = 1.0 + (atoi(strZoomFactor.c_str()) / 100.0);
  }
  // allwaysReact
  std::string strAlwaysReact = "";
  if (properties->GetStringProperty("alwaysReact", strAlwaysReact))
  {
    if (strAlwaysReact == "true")
    {
      m_AlwaysReact = true;
    }
    else
    {
      m_AlwaysReact = false;
    }
  }
  else
  {
    m_AlwaysReact = false;
  }
}

bool mitk::MIDASDisplayInteractor::FilterEvents(InteractionEvent* /*interactionEvent*/, DataNode* /*dataNode*/)
{
  return true;
}

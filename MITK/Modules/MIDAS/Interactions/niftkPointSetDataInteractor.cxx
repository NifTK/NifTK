/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPointSetDataInteractor.h"

#include <mitkBaseRenderer.h>
#include <mitkInteractionConst.h>
#include <mitkInteractionPositionEvent.h>
#include <mitkPointSet.h>
#include <mitkPositionEvent.h>
#include <mitkRenderingManager.h>

namespace niftk
{

//-----------------------------------------------------------------------------
PointSetDataInteractor::PointSetDataInteractor()
: mitk::PointSetDataInteractor()
{
  /// We can set a very high accuracy here, since sub-voxel positions are rounded
  /// to the voxel centre positions.
  this->SetAccuracy(0.001);
}


//-----------------------------------------------------------------------------
PointSetDataInteractor::~PointSetDataInteractor()
{
}


//-----------------------------------------------------------------------------
bool PointSetDataInteractor::FilterEvents(mitk::InteractionEvent* event, mitk::DataNode* dataNode)
{
  return FilteringStateMachine::CanHandleEvent(event);
}


//-----------------------------------------------------------------------------
bool PointSetDataInteractor::CheckCondition(const mitk::StateMachineCondition& condition, const mitk::InteractionEvent* event)
{
  const mitk::InteractionPositionEvent* positionEvent =
      dynamic_cast<const mitk::InteractionPositionEvent*>(event);

  if (positionEvent)
  {
    /// We replace the sub-voxel position by the voxel centre position.

    mitk::BaseRenderer* renderer = positionEvent->GetSender();

    mitk::Point3D point3DInMm = positionEvent->GetPositionInWorld();
    const mitk::BaseGeometry* worldGeometry = renderer->GetWorldGeometry();
    mitk::Point3D point3DIndex;
    worldGeometry->WorldToIndex(point3DInMm, point3DIndex);
    point3DIndex[0] = std::floor(point3DIndex[0]) + 0.5;
    point3DIndex[1] = std::floor(point3DIndex[1]) + 0.5;
    worldGeometry->IndexToWorld(point3DIndex, point3DInMm);

    mitk::Point2D point2DInMm;
    mitk::Point2D point2DInPx;

    mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();
    displayGeometry->Map(point3DInMm, point2DInMm);
    displayGeometry->WorldToDisplay(point2DInMm, point2DInPx);

    mitk::InteractionPositionEvent::Pointer positionEvent2 =
        mitk::InteractionPositionEvent::New(renderer, point2DInPx, point3DInMm);

    return Superclass::CheckCondition(condition, positionEvent2.GetPointer());
  }

  return Superclass::CheckCondition(condition, event);
}


//-----------------------------------------------------------------------------
bool PointSetDataInteractor::ExecuteAction(mitk::StateMachineAction* action, mitk::InteractionEvent* event)
{
  mitk::InteractionPositionEvent* positionEvent =
      dynamic_cast<mitk::InteractionPositionEvent*>(event);

  if (positionEvent)
  {
    /// We replace the sub-voxel position by the voxel centre position.

    mitk::BaseRenderer* renderer = positionEvent->GetSender();

    mitk::Point3D point3DInMm = positionEvent->GetPositionInWorld();
    const mitk::BaseGeometry* worldGeometry = renderer->GetWorldGeometry();
    mitk::Point3D point3DIndex;
    worldGeometry->WorldToIndex(point3DInMm, point3DIndex);
    point3DIndex[0] = std::floor(point3DIndex[0]) + 0.5;
    point3DIndex[1] = std::floor(point3DIndex[1]) + 0.5;
    worldGeometry->IndexToWorld(point3DIndex, point3DInMm);

    mitk::Point2D point2DInMm;
    mitk::Point2D point2DInPx;

    mitk::DisplayGeometry* displayGeometry = renderer->GetDisplayGeometry();
    displayGeometry->Map(point3DInMm, point2DInMm);
    displayGeometry->WorldToDisplay(point2DInMm, point2DInPx);

    mitk::InteractionPositionEvent::Pointer positionEvent2 =
        mitk::InteractionPositionEvent::New(renderer, point2DInPx, point3DInMm);

    return Superclass::ExecuteAction(action, positionEvent2.GetPointer());
  }

  return Superclass::ExecuteAction(action, event);
}

}

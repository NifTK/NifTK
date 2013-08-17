/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTrackedPointerManager.h"
#include <mitkCoordinateAxesData.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <mitkSurface.h>
#include <mitkTimeSlicedGeometry.h>
#include <mitkRenderingManager.h>
#include <mitkPointSetUpdate.h>
#include <mitkPointUtils.h>
#include <mitkUndoController.h>
#include <mitkOperation.h>
#include <mitkOperationActor.h>

const bool mitk::TrackedPointerManager::UPDATE_VIEW_COORDINATE_DEFAULT(false);
const std::string mitk::TrackedPointerManager::TRACKED_POINTER_POINTSET_NAME("TrackedPointerManagerPointSet");
const mitk::OperationType mitk::TrackedPointerManager::OP_UPDATE_POINTSET(9034657);

namespace mitk
{

//-----------------------------------------------------------------------------
TrackedPointerManager::TrackedPointerManager()
{
}


//-----------------------------------------------------------------------------
TrackedPointerManager::~TrackedPointerManager()
{
  if (m_DataStorage.IsNotNull())
  {
    mitk::DataNode::Pointer pointSetNode = m_DataStorage->GetNamedNode(TRACKED_POINTER_POINTSET_NAME);
    if (pointSetNode.IsNull())
    {
      m_DataStorage->Remove(pointSetNode);
    }
  }
}


//-----------------------------------------------------------------------------
void TrackedPointerManager::SetDataStorage(const mitk::DataStorage::Pointer& storage)
{
  m_DataStorage = storage;
  this->Modified();
}


//-----------------------------------------------------------------------------
mitk::PointSet::Pointer TrackedPointerManager::RetrievePointSet()
{
  assert(m_DataStorage);
  mitk::PointSet::Pointer result = NULL;

  mitk::DataNode::Pointer pointSetNode = m_DataStorage->GetNamedNode(TRACKED_POINTER_POINTSET_NAME);
  if (pointSetNode.IsNull())
  {
    result = mitk::PointSet::New();

    pointSetNode = mitk::DataNode::New();
    pointSetNode->SetData(result);
    pointSetNode->SetName(TRACKED_POINTER_POINTSET_NAME);
    m_DataStorage->Add(pointSetNode);
  }
  else
  {
    result = dynamic_cast<mitk::PointSet*>(pointSetNode->GetData());
  }

  return result;
}


//-----------------------------------------------------------------------------
void TrackedPointerManager::OnGrabPoint(const mitk::Point3D& point)
{
  mitk::PointSet::Pointer currentPointSet = this->RetrievePointSet();

  mitk::PointSetUpdate* doOp = new mitk::PointSetUpdate(OP_UPDATE_POINTSET, currentPointSet);
  doOp->AppendPoint(point);

  mitk::PointSetUpdate* undoOp = new mitk::PointSetUpdate(OP_UPDATE_POINTSET, currentPointSet);
  mitk::OperationEvent *operationEvent = new mitk::OperationEvent(this, doOp, undoOp, "Update PointSet");
  mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );

  this->ExecuteOperation(doOp);
}


//-----------------------------------------------------------------------------
void TrackedPointerManager::OnClearPoints()
{
  mitk::PointSet::Pointer currentPointSet = this->RetrievePointSet();

  mitk::PointSetUpdate* doOp = new mitk::PointSetUpdate(OP_UPDATE_POINTSET, NULL);
  mitk::PointSetUpdate* undoOp = new mitk::PointSetUpdate(OP_UPDATE_POINTSET, currentPointSet);
  mitk::OperationEvent *operationEvent = new mitk::OperationEvent(this, doOp, undoOp, "Update PointSet");
  mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent( operationEvent );

  this->ExecuteOperation(doOp);
}


//-----------------------------------------------------------------------------
void TrackedPointerManager::ExecuteOperation(mitk::Operation* operation)
{
  assert(m_DataStorage);
  assert(operation);

  switch (operation->GetOperationType())
  {
    case OP_UPDATE_POINTSET:

      mitk::PointSetUpdate* op = static_cast<mitk::PointSetUpdate*>(operation);
      mitk::PointSet::Pointer pointSet = this->RetrievePointSet();
      const mitk::PointSet *newPointSet = op->GetPointSet();

      mitk::CopyPointSets(*newPointSet,  *pointSet);

      mitk::RenderingManager::GetInstance()->RequestUpdateAll();

    break;
  }
}


//-----------------------------------------------------------------------------
void TrackedPointerManager::Update(
         const vtkMatrix4x4* tipToPointerTransform,
         const mitk::DataNode::Pointer pointerToWorldNode,
         const mitk::DataNode::Pointer probeModel,
         mitk::Point3D& tipCoordinate
         )
{
  if (tipToPointerTransform == NULL)
  {
    MITK_ERROR << "TrackedPointerManager::Update, invalid tipToPointerTransform";
    return;
  }

  mitk::CoordinateAxesData::Pointer pointerToWorld = dynamic_cast<mitk::CoordinateAxesData*>(pointerToWorldNode->GetData());
  if (pointerToWorld.IsNull())
  {
    MITK_ERROR << "TrackedPointerManager::Update, invalid pointerToWorldNode";
    return;
  }

  vtkSmartPointer<vtkMatrix4x4> pointerToWorldTransform = vtkMatrix4x4::New();
  pointerToWorld->GetVtkMatrix(*pointerToWorldTransform);

  vtkSmartPointer<vtkMatrix4x4> combinedTransform = vtkMatrix4x4::New();
  combinedTransform->Identity();

  combinedTransform->Multiply4x4(pointerToWorldTransform, tipToPointerTransform, combinedTransform);

  if (probeModel.IsNotNull())
  {
    mitk::BaseData::Pointer model = dynamic_cast<mitk::BaseData*>(probeModel->GetData());
    if (model.IsNotNull())
    {
      mitk::Geometry3D::Pointer geometry = model->GetGeometry();
      if (geometry.IsNotNull())
      {
        geometry->SetIndexToWorldTransformByVtkMatrix(combinedTransform);
        geometry->Modified();
      }
      model->Modified();
      probeModel->Modified();
    }
  }

  double coordinateIn[4] = {0, 0, 0, 1};
  double coordinateOut[4] = {0, 0, 0, 1};

  coordinateIn[0] = tipCoordinate[0];
  coordinateIn[1] = tipCoordinate[1];
  coordinateIn[2] = tipCoordinate[2];

  combinedTransform->MultiplyPoint(coordinateIn, coordinateOut);

  tipCoordinate[0] = coordinateOut[0];
  tipCoordinate[1] = coordinateOut[1];
  tipCoordinate[2] = coordinateOut[2];
}

//-----------------------------------------------------------------------------
} // end namespace


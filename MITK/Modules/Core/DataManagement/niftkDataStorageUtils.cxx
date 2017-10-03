/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkDataStorageUtils.h"

#include <mitkApplyTransformMatrixOperation.h>
#include <mitkDataStorage.h>
#include <mitkInteractionConst.h>
#include <mitkNodePredicateDataType.h>
#include <mitkOperationEvent.h>
#include <mitkUndoController.h>

#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>

#include "niftkAffineTransformDataNodeProperty.h"
#include "niftkCoordinateAxesData.h"
#include "niftkFileIOUtils.h"

namespace niftk
{

//-----------------------------------------------------------------------------
bool IsNodeABinaryImage(const mitk::DataNode* node)
{
  bool isBinary;

  return node
      && dynamic_cast<mitk::Image*>(node->GetData())
      && node->GetBoolProperty("binary", isBinary)
      && isBinary;
}


//-----------------------------------------------------------------------------
bool IsNodeAnUcharBinaryImage(const mitk::DataNode* node)
{
  return IsNodeABinaryImage(node)
      && dynamic_cast<mitk::Image*>(node->GetData())->GetPixelType().GetComponentType() == itk::ImageIOBase::UCHAR;
}


//-----------------------------------------------------------------------------
bool IsNodeANonBinaryImage(const mitk::DataNode* node)
{
  bool isBinary;

  return node
      && dynamic_cast<mitk::Image*>(node->GetData())
      && (!node->GetBoolProperty("binary", isBinary) || !isBinary);
}


//-----------------------------------------------------------------------------
bool IsNodeAGreyScaleImage(const mitk::DataNode* node)
{
  return IsNodeANonBinaryImage(node)
      && dynamic_cast<mitk::Image*>(node->GetData())->GetPixelType().GetNumberOfComponents() == 1;
}


//-----------------------------------------------------------------------------
bool IsNodeAHelperObject(const mitk::DataNode* node)
{
  bool result = false;

  if (node)
  {
    bool isHelper;
    if (node->GetBoolProperty("helper object", isHelper) && isHelper)
    {
      result = true;
    }
  }

  return result;
}


//-----------------------------------------------------------------------------
mitk::DataNode* FindFirstParentImage(const mitk::DataStorage* storage, const mitk::DataNode* node, bool lookForBinary)
{
  mitk::DataNode* result = nullptr;

  mitk::TNodePredicateDataType<mitk::Image>::Pointer isImage = mitk::TNodePredicateDataType<mitk::Image>::New();
  mitk::DataStorage::SetOfObjects::ConstPointer possibleParents = storage->GetSources(node, isImage);

  for (unsigned int i = 0; i < possibleParents->size(); i++)
  {
    mitk::DataNode* possibleNode = (*possibleParents)[i];

    bool isBinary = false;
    possibleNode->GetBoolProperty("binary", isBinary);

    if (isBinary == lookForBinary)
    {
      result = possibleNode;
    }
  }

  return result;
}


//-----------------------------------------------------------------------------
mitk::DataStorage::SetOfObjects::Pointer FindNodesStartingWith(const mitk::DataStorage* dataStorage, const std::string prefix)
{
  mitk::DataStorage::SetOfObjects::Pointer results = mitk::DataStorage::SetOfObjects::New();

  unsigned int counter = 0;
  mitk::DataStorage::SetOfObjects::ConstPointer all = dataStorage->GetAll();
  for (unsigned int i = 0; i < all->size(); i++)
  {
    mitk::DataNode* possibleNode = (*all)[i];
    if (possibleNode->GetName().compare(0, prefix.length(), prefix) == 0)
    {
      results->InsertElement(counter, possibleNode);
      counter++;
    }
  }

  return results;
}


//-----------------------------------------------------------------------------
void LoadMatrixOrCreateDefault(
    const std::string& fileName,
    const std::string& nodeName,
    bool helperObject,
    mitk::DataStorage* dataStorage)
{
  vtkSmartPointer<vtkMatrix4x4> matrix = LoadVtkMatrix4x4FromFile(fileName);

  mitk::DataNode::Pointer node = dataStorage->GetNamedNode(nodeName);
  if (node.IsNull())
  {
    node = mitk::DataNode::New();
    node->SetName(nodeName);
    node->SetBoolProperty("show text", false);
    node->SetIntProperty("size", 10);
    node->SetVisibility(false); // by default we don't need to see it.
    node->SetBoolProperty("helper object", helperObject);
  }

  std::string propertyName = "niftk.transform";
  AffineTransformDataNodeProperty::Pointer affTransProp = static_cast<AffineTransformDataNodeProperty*>(node->GetProperty(propertyName.c_str()));
  if (affTransProp.IsNull())
  {
    affTransProp = AffineTransformDataNodeProperty::New();
    node->SetProperty(propertyName.c_str(), affTransProp);
  }
  affTransProp->SetTransform(*matrix);

  CoordinateAxesData::Pointer coordinateAxes = dynamic_cast<CoordinateAxesData*>(node->GetData());
  if (coordinateAxes.IsNull())
  {
    coordinateAxes = CoordinateAxesData::New();
    node->SetData(coordinateAxes);
  }
  coordinateAxes->SetVtkMatrix(*matrix);

  if (!dataStorage->Exists(node))
  {
    dataStorage->Add(node);
  }
  node->Modified();
}


//-----------------------------------------------------------------------------
void GetCurrentTransformFromNode(const mitk::DataNode::Pointer& node , vtkMatrix4x4& outputMatrix)
{
  if (node.IsNull())
  {
    mitkThrow() << "In GetCurrentTransformFromNode, node is nullptr";
  }

  mitk::AffineTransform3D::Pointer affineTransform = node->GetData()->GetGeometry()->GetIndexToWorldTransform();
  itk::Matrix<double, 3, 3>  matrix;
  itk::Vector<double, 3> offset;
  matrix = affineTransform->GetMatrix();
  offset = affineTransform->GetOffset();

  outputMatrix.Identity();
  for ( int i = 0 ; i < 3 ; i ++ )
  {
    for ( int j = 0 ; j < 3 ; j ++ )
    {
      outputMatrix.SetElement (i, j, matrix[i][j]);
    }
  }
  for ( int i = 0 ; i < 3 ; i ++ )
  {
    outputMatrix.SetElement (i, 3, offset[i]);
  }
}


//-----------------------------------------------------------------------------
void ComposeTransformWithNode(const vtkMatrix4x4& transform, mitk::DataNode::Pointer& node)
{
  if (node.IsNull())
  {
    mitkThrow() << "In ComposeTransformWithNode, node is nullptr";
  }

  vtkSmartPointer<vtkMatrix4x4> currentMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  GetCurrentTransformFromNode(node, *currentMatrix);

  vtkSmartPointer<vtkMatrix4x4> newMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
  newMatrix->Multiply4x4(&transform, currentMatrix, newMatrix);

  ApplyTransformToNode(*newMatrix, node);
}


//-----------------------------------------------------------------------------
void ApplyTransformToNode(const vtkMatrix4x4& transform, mitk::DataNode::Pointer& node)
{
  if (node.IsNull())
  {
    mitkThrow() << "In ApplyTransformToNode, node is nullptr";
  }

  mitk::BaseData::Pointer baseData = node->GetData();
  if (baseData.IsNull())
  {
    mitkThrow() << "In ApplyTransformToNode, baseData is nullptr";
  }

  mitk::BaseGeometry* geometry = baseData->GetGeometry();
  if (!geometry)
  {
    mitkThrow() << "In ApplyTransformToNode, geometry is nullptr";
  }

  CoordinateAxesData::Pointer axes = dynamic_cast<CoordinateAxesData*>(node->GetData());
  if (axes.IsNotNull())
  {
    AffineTransformDataNodeProperty::Pointer property = dynamic_cast<AffineTransformDataNodeProperty*>(node->GetProperty("niftk.transform"));
    if (property.IsNotNull())
    {
      property->SetTransform(transform);
      property->Modified();
    }
  }

  vtkSmartPointer<vtkMatrix4x4> nonConstTransform = vtkSmartPointer<vtkMatrix4x4>::New();
  nonConstTransform->DeepCopy(&transform);

  mitk::Point3D dummyPoint;
  dummyPoint.Fill(0);

  mitk::ApplyTransformMatrixOperation* doOp = new mitk::ApplyTransformMatrixOperation(mitk::OpAPPLYTRANSFORMMATRIX, nonConstTransform, dummyPoint);

  if (mitk::UndoController::GetCurrentUndoModel())
  {
    vtkSmartPointer<vtkMatrix4x4> inverse = vtkSmartPointer<vtkMatrix4x4>::New();
    inverse->DeepCopy(&transform);
    inverse->Invert();

    mitk::ApplyTransformMatrixOperation* undoOp = new mitk::ApplyTransformMatrixOperation(mitk::OpAPPLYTRANSFORMMATRIX, inverse, dummyPoint);
    mitk::OperationEvent* operationEvent = new mitk::OperationEvent(geometry, doOp, undoOp, "ApplyTransformToNode");
    mitk::UndoController::GetCurrentUndoModel()->SetOperationEvent(operationEvent);
    geometry->ExecuteOperation(doOp);
  }
  else
  {
    geometry->ExecuteOperation(doOp);
    delete doOp;
  }
}

}

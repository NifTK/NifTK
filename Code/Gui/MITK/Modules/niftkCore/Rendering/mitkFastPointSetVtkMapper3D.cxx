/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkFastPointSetVtkMapper3D.h"
#include <mitkDataNode.h>
#include <mitkProperties.h>
#include <mitkColorProperty.h>
#include <mitkVtkPropRenderer.h>
#include <mitkPointSet.h>
#include <mitkVector.h>

#include <vtkIdTypeArray.h>
#include <vtkFloatArray.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>

namespace mitk {

//-----------------------------------------------------------------------------
FastPointSetVtkMapper3D::FastPointSetVtkMapper3D()
: m_NumberOfPoints(0)
{
}


//-----------------------------------------------------------------------------
FastPointSetVtkMapper3D::~FastPointSetVtkMapper3D()
{
}


//-----------------------------------------------------------------------------
FastPointSetVtkMapper3D::LocalStorage::LocalStorage()
: m_Indicies(NULL)
, m_Array(NULL)
, m_Points(NULL)
, m_CellArray(NULL)
, m_PolyData(NULL)
, m_PolyDataMapper(NULL)
, m_Actor(NULL)
{
  m_Indicies = vtkIdTypeArray::New();
  m_Array = vtkFloatArray::New();
  m_Points = vtkPoints::New();
  m_CellArray = vtkCellArray::New();
  m_PolyData = vtkPolyData::New();
  m_PolyDataMapper = vtkPolyDataMapper::New();
  m_Actor = vtkActor::New();
}


//-----------------------------------------------------------------------------
FastPointSetVtkMapper3D::LocalStorage::~LocalStorage()
{
}


//-----------------------------------------------------------------------------
const mitk::PointSet* FastPointSetVtkMapper3D::GetInput()
{
  return static_cast<const mitk::PointSet * > ( GetDataNode()->GetData() );
}


//-----------------------------------------------------------------------------
vtkProp* FastPointSetVtkMapper3D::GetVtkProp(mitk::BaseRenderer *renderer)
{
  LocalStorage *ls = m_LocalStorage.GetLocalStorage(renderer);
  return ls->m_Actor;
}


//-----------------------------------------------------------------------------
void FastPointSetVtkMapper3D::ResetMapper( mitk::BaseRenderer* renderer )
{
  LocalStorage *ls = m_LocalStorage.GetLocalStorage(renderer);
  ls->m_Actor->VisibilityOff();
}


//-----------------------------------------------------------------------------
void FastPointSetVtkMapper3D::GenerateDataForRenderer(mitk::BaseRenderer* renderer)
{
  LocalStorage *ls = m_LocalStorage.GetLocalStorage(renderer);

  mitk::DataNode* dataNode = this->GetDataNode();
  assert(dataNode);

  bool visible = true;
  bool gotVisibility = dataNode->GetVisibility(visible, renderer);
  if (!gotVisibility)
  {
    dataNode->SetVisibility(visible, renderer);
  }

  if(!visible)
  {
    ls->m_Actor->VisibilityOff();
    return;
  }
  else
  {
    ls->m_Actor->VisibilityOn();
  }

  float opacity = 1;
  bool gotOpacity = dataNode->GetOpacity(opacity, renderer);
  if (!gotOpacity)
  {
    dataNode->SetBoolProperty("opacity", opacity, renderer);
  }
  ls->m_Actor->GetProperty()->SetOpacity(opacity);

  int pointSize = 1;
  bool gotPointSize = dataNode->GetIntProperty("pointsize", pointSize);
  if (!gotPointSize)
  {
    dataNode->SetIntProperty("pointsize", pointSize, renderer);
  }
  ls->m_Actor->GetProperty()->SetPointSize(pointSize);

  if (dynamic_cast<mitk::ColorProperty*>(dataNode->GetPropertyList(renderer)->GetProperty("color")) == NULL)
  {
    float unselectedColor[3] = {1.0f, 1.0f, 0.0f};
    dataNode->SetColor(unselectedColor, renderer, "color");
  }

  if (dynamic_cast<mitk::ColorProperty*>(dataNode->GetPropertyList(renderer)->GetProperty("selectedcolor")) == NULL)
  {
    float selectedColor[3] = {1.0f, 0.0f, 0.0f};
    dataNode->SetColor(selectedColor, renderer, "selectedcolor");
  }

  mitk::Color colour;
  vtkFloatingPointType colourVtk[4]={1.0f, 1.0f, 1.0f,1.0f};

  bool isSelected = false;
  bool gotSelected = dataNode->GetBoolProperty("selected", isSelected);
  if (!gotSelected)
  {
    dataNode->SetBoolProperty("selected", isSelected, renderer);
  }

  if (isSelected)
  {
    colour = dynamic_cast<mitk::ColorProperty *>(this->GetDataNode()->GetPropertyList(renderer)->GetProperty("selectedcolor"))->GetValue();
  }
  else
  {
    colour = dynamic_cast<mitk::ColorProperty *>(this->GetDataNode()->GetPropertyList(renderer)->GetProperty("color"))->GetValue();
  }
  colourVtk[0] = colour[0];
  colourVtk[1] = colour[1];
  colourVtk[2] = colour[2];
  ls->m_Actor->GetProperty()->SetColor(colourVtk);

  mitk::PointSet::PointType point;
  const mitk::PointSet* pointSet = this->GetInput();
  mitk::PointSet::DataType* itkPointSet = pointSet->GetPointSet( 0 );
  mitk::PointSet::PointsContainer* points = itkPointSet->GetPoints();

  if (pointSet != NULL && pointSet->GetSize() > 0)
  {
    unsigned long int numberOfPoints = pointSet->GetSize();

    // Only allocate new if the number of points have changed.
    if (ls->m_Points == NULL || m_NumberOfPoints != numberOfPoints)
    {
      ls->m_Indicies->SetNumberOfComponents(1);
      ls->m_Indicies->SetNumberOfValues(numberOfPoints*2);
      ls->m_Array->SetNumberOfComponents(3);
      ls->m_Array->SetNumberOfValues(numberOfPoints*3);
      ls->m_Points->SetData(ls->m_Array);
      ls->m_CellArray->SetCells(numberOfPoints, ls->m_Indicies);
      ls->m_PolyData->SetPoints(ls->m_Points);
      ls->m_PolyData->SetVerts(ls->m_CellArray);
      ls->m_PolyDataMapper->SetInputConnection(0, ls->m_PolyData->GetProducerPort());
      ls->m_Actor->SetMapper(ls->m_PolyDataMapper);
      m_NumberOfPoints = numberOfPoints;
    }

    unsigned long int pointCounter = 0;
    unsigned long int arrayCounter = 0;
    unsigned long int indexCounter = 0;
    mitk::PointSet::PointsIterator pIt;

    // Processing each point one at a time.
    for (pIt = points->Begin(); pIt != points->End(); ++pIt)  // for each point in the pointset
    {
      point = pIt->Value();

      arrayCounter = pointCounter*3;
      indexCounter = pointCounter*2;

      // Copy x,y,z in sequence for each point.
      ls->m_Array->SetValue(arrayCounter, point[0]);
      ls->m_Array->SetValue(arrayCounter+1, point[1]);
      ls->m_Array->SetValue(arrayCounter+2, point[2]);

      // Copy a list of cells, where the first number is
      // the number of points in the cell (always 1), and
      // the second number is the point ID.
      ls->m_Indicies->SetValue(indexCounter, 1);
      ls->m_Indicies->SetValue(indexCounter+1, pointCounter);

      pointCounter += 1;
    }
  }
}

//-----------------------------------------------------------------------------
} // end namespace


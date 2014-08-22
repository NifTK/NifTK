/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkImage2DToTexturePlaneMapper3D.h"
#include <mitkDataNode.h>
#include <mitkProperties.h>
#include <mitkColorProperty.h>
#include <mitkVtkPropRenderer.h>
#include <mitkImage.h>
#include <mitkVector.h>
#include <mitkExceptionMacro.h>
#include <mitkPointUtils.h>

#include <vtkIdTypeArray.h>
#include <vtkFloatArray.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkPointData.h>

namespace mitk {

//-----------------------------------------------------------------------------
Image2DToTexturePlaneMapper3D::Image2DToTexturePlaneMapper3D()
{
}


//-----------------------------------------------------------------------------
Image2DToTexturePlaneMapper3D::~Image2DToTexturePlaneMapper3D()
{
}


//-----------------------------------------------------------------------------
Image2DToTexturePlaneMapper3D::LocalStorage::LocalStorage()
: m_PointArray(NULL)
, m_TextureArray(NULL)
, m_NormalsArray(NULL)
, m_Points(NULL)
, m_CellArray(NULL)
, m_PolyData(NULL)
, m_Texture(NULL)
, m_PolyDataMapper(NULL)
, m_Actor(NULL)
, m_NumberOfPoints(0)
{
  m_TextureArray = vtkFloatArray::New();
  m_TextureArray->SetNumberOfComponents(3);
  m_TextureArray->SetNumberOfTuples(4);
  m_TextureArray->SetTuple3(0, 0, 0, 0);
  m_TextureArray->SetTuple3(1, 1, 0, 0);
  m_TextureArray->SetTuple3(2, 1, 1, 0);
  m_TextureArray->SetTuple3(3, 0, 1, 0);

  m_NormalsArray = vtkFloatArray::New();
  m_NormalsArray->SetNumberOfComponents(3);
  m_NormalsArray->SetNumberOfTuples(4);
  m_NormalsArray->SetTuple3(0, 0, 0, 1);
  m_NormalsArray->SetTuple3(1, 0, 0, 1);
  m_NormalsArray->SetTuple3(2, 0, 0, 1);
  m_NormalsArray->SetTuple3(3, 0, 0, 1);

  m_PointArray = vtkFloatArray::New();
  m_PointArray->SetNumberOfComponents(3);
  m_PointArray->SetNumberOfTuples(4);

  m_Points = vtkPoints::New();
  m_Points->SetData(m_PointArray);

  m_CellArray = vtkCellArray::New();
  m_CellArray->InsertNextCell(4);
  m_CellArray->InsertCellPoint(0);
  m_CellArray->InsertCellPoint(1);
  m_CellArray->InsertCellPoint(2);
  m_CellArray->InsertCellPoint(3);

  m_PolyData = vtkPolyData::New();
  m_PolyData->SetPoints(m_Points);
  m_PolyData->SetPolys(m_CellArray);
  m_PolyData->GetPointData()->SetTCoords(m_TextureArray);
  m_PolyData->GetPointData()->SetNormals(m_NormalsArray);

  m_Texture = vtkTexture::New();

  m_PolyDataMapper = vtkPolyDataMapper::New();
  m_PolyDataMapper->SetInputData(m_PolyData);

  m_Actor = vtkActor::New();
  m_Actor->SetMapper(m_PolyDataMapper);
  m_Actor->SetTexture(m_Texture);
}


//-----------------------------------------------------------------------------
Image2DToTexturePlaneMapper3D::LocalStorage::~LocalStorage()
{
}


//-----------------------------------------------------------------------------
const mitk::Image* Image2DToTexturePlaneMapper3D::GetInput()
{
  return static_cast<const mitk::Image*> ( GetDataNode()->GetData() );
}


//-----------------------------------------------------------------------------
vtkProp* Image2DToTexturePlaneMapper3D::GetVtkProp(mitk::BaseRenderer *renderer)
{
  LocalStorage *ls = m_LocalStorage.GetLocalStorage(renderer);
  return ls->m_Actor;
}


//-----------------------------------------------------------------------------
void Image2DToTexturePlaneMapper3D::ResetMapper( mitk::BaseRenderer* renderer )
{
  LocalStorage *ls = m_LocalStorage.GetLocalStorage(renderer);
  ls->m_Actor->VisibilityOff();
}


//-----------------------------------------------------------------------------
void Image2DToTexturePlaneMapper3D::GenerateDataForRenderer(mitk::BaseRenderer* renderer)
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
  ls->m_Actor->GetProperty()->SetInterpolationToFlat();
  ls->m_Actor->GetProperty()->SetLighting(false);

  mitk::Point3D indexPoints[5];
  mitk::Point3D worldPoints[5];
  mitk::Point3D normal;


  mitk::Image* image = const_cast<mitk::Image*>(this->GetInput());

  if (image != NULL && image->GetNumberOfChannels() == 1 && image->GetDimension() == 2)
  {
    indexPoints[0][0] = -0.5;
    indexPoints[0][1] = -0.5;
    indexPoints[0][2] = 0;
    indexPoints[1][0] = image->GetDimension(0) - 0.5;
    indexPoints[1][1] = -0.5;
    indexPoints[1][2] = 0;
    indexPoints[2][0] = image->GetDimension(0) - 0.5;
    indexPoints[2][1] = image->GetDimension(1) - 0.5;
    indexPoints[2][2] = 0;
    indexPoints[3][0] = -0.5;
    indexPoints[3][1] = image->GetDimension(1) - 0.5;
    indexPoints[3][2] = 0;

    // For surface normal
    indexPoints[4][0] = -0.5;
    indexPoints[4][1] = -0.5;
    indexPoints[4][2] = -1;

    for (unsigned int i = 0; i < 4; i++)
    {
      image->GetGeometry()->IndexToWorld(indexPoints[i], worldPoints[i]);
      ls->m_PointArray->SetTuple3(i, worldPoints[i][0], worldPoints[i][1], worldPoints[i][2]);
    }

    image->GetGeometry()->IndexToWorld(indexPoints[4], worldPoints[4]);
    mitk::GetDifference(worldPoints[4], worldPoints[0], normal);
    mitk::Normalise(normal);
    for (unsigned int i = 0; i < 4; i++)
    {
      ls->m_NormalsArray->SetTuple3(i, normal[0], normal[1], normal[2]);
    }

    ls->m_Texture->SetInputData(image->GetVtkImageData());
  }
  else
  {
    mitkThrow() << "Image2DToTexturePlaneMapper3D assigned to invalid image." << std::endl;
  }
}

//-----------------------------------------------------------------------------
} // end namespace


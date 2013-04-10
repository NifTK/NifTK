/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkCoordinateAxesVtkMapper3D.h"
#include <vtkAxesActor.h>
#include <vtkPolyDataMapper.h>

namespace mitk {

//-----------------------------------------------------------------------------
CoordinateAxesVtkMapper3D::CoordinateAxesVtkMapper3D()
{
}


//-----------------------------------------------------------------------------
CoordinateAxesVtkMapper3D::~CoordinateAxesVtkMapper3D()
{
}


//-----------------------------------------------------------------------------
const CoordinateAxesData* CoordinateAxesVtkMapper3D::GetInput()
{
  return static_cast<const mitk::CoordinateAxesData * > ( this->GetDataNode()->GetData() );
}


//-----------------------------------------------------------------------------
vtkProp* CoordinateAxesVtkMapper3D::GetVtkProp(mitk::BaseRenderer *renderer)
{
  LocalStorage *ls = m_LocalStorage.GetLocalStorage(renderer);
  return ls->m_Actor;
}


//-----------------------------------------------------------------------------
void CoordinateAxesVtkMapper3D::ResetMapper( mitk::BaseRenderer* renderer )
{
  LocalStorage *ls = m_LocalStorage.GetLocalStorage(renderer);
  ls->m_Actor->VisibilityOff();
}


//-----------------------------------------------------------------------------
void CoordinateAxesVtkMapper3D::GenerateDataForRenderer(mitk::BaseRenderer* renderer)
{
  LocalStorage *ls = m_LocalStorage.GetLocalStorage(renderer);

  bool visible = true;
  this->GetDataNode()->GetVisibility(visible, renderer, "visible");

  if(!visible)
  {
    ls->m_Actor->VisibilityOff();
    return;
  }

  // Update the transformation to match that on the node, which should be a mitkCoordinateAxesData.
  mitk::CoordinateAxesData::Pointer axesData = dynamic_cast<mitk::CoordinateAxesData*>(this->GetDataNode()->GetData());
  if (axesData.IsNotNull())
  {
    vtkMatrix4x4 *matrix = vtkMatrix4x4::New();
    axesData->GetVtkMatrix(*matrix);
    ls->m_Actor->SetUserMatrix(matrix);
    matrix->Delete();
  }

}


//-----------------------------------------------------------------------------
CoordinateAxesVtkMapper3D::LocalStorage::LocalStorage()
  : m_Actor(NULL)
{
  m_Actor = vtkAxesActor::New();

  // Basic configuration
  m_Actor->SetShaftTypeToLine();
  m_Actor->SetXAxisLabelText("x");
  m_Actor->SetYAxisLabelText("y");
  m_Actor->SetZAxisLabelText("z");
  m_Actor->AxisLabelsOn();
}


//-----------------------------------------------------------------------------
CoordinateAxesVtkMapper3D::LocalStorage::~LocalStorage()
{
  m_Actor->Delete();
}

//-----------------------------------------------------------------------------
} // end namespace

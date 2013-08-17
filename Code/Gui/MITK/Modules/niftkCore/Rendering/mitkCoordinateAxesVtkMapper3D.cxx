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
void CoordinateAxesVtkMapper3D::SetDefaultProperties(mitk::DataNode* node, mitk::BaseRenderer* renderer, bool overwrite)
{
  // Temporary: By default, start off invisible, as there is a current glitch
  // which means that with visibility on, the OverlayEditor goes nuts.
  node->AddProperty( "visible", mitk::BoolProperty::New(false), renderer, overwrite );

  // However: This seems to help. I haven't had time to test taking the above bit out!
  node->AddProperty( "includeInBoundingBox", mitk::BoolProperty::New(false), renderer, overwrite );

  // Hereon ... business as usual.
  node->AddProperty( "show text", mitk::BoolProperty::New(false), renderer, overwrite );
  node->AddProperty( "size", mitk::IntProperty::New(10), renderer, overwrite );
  Superclass::SetDefaultProperties(node, renderer, overwrite);
}


//-----------------------------------------------------------------------------
void CoordinateAxesVtkMapper3D::GenerateDataForRenderer(mitk::BaseRenderer* renderer)
{
  LocalStorage *ls = m_LocalStorage.GetLocalStorage(renderer);

  bool visible = true;
  this->GetDataNode()->GetVisibility(visible, renderer, "visible");

  bool showText = false;
  this->GetDataNode()->GetBoolProperty("show text", showText, renderer );

  int length = 10; // millimetres
  this->GetDataNode()->GetIntProperty("size", length, renderer);

  if(!visible)
  {
    ls->m_Actor->VisibilityOff();
    return;
  }
  else
  {
    ls->m_Actor->VisibilityOn();
  }

  if(showText)
  {
    ls->m_Actor->AxisLabelsOn();
  }
  else
  {
    ls->m_Actor->AxisLabelsOff();
  }

  ls->m_Actor->SetTotalLength(length, length, length);
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

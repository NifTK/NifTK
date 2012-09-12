/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $LastChangedDate: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : $Author: mjc $

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkIGITestDataUtils.h"
#include <mitkCone.h>
#include <mitkSurface.h>
#include <vtkConeSource.h>
#include <QFile>

namespace mitk
{

//-----------------------------------------------------------------------------
mitk::DataNode::Pointer CreateConeRepresentation(
    const char* label,
    mitk::Vector3D& centerPoint,
    mitk::Vector3D& direction
    )
{
  // new data
  vtkConeSource* vtkData = vtkConeSource::New();
  vtkData->SetRadius(7.5);
  vtkData->SetHeight(15.0);
  vtkData->SetDirection(direction[0], direction[1], direction[2]);
  vtkData->SetCenter(centerPoint[0], centerPoint[1], centerPoint[2]);
  vtkData->SetResolution(20);
  vtkData->CappingOn();
  vtkData->Update();

  mitk::Cone::Pointer activeToolData = mitk::Cone::New();
  activeToolData->SetVtkPolyData(vtkData->GetOutput());
  vtkData->Delete();

  // new node
  mitk::DataNode::Pointer coneNode = mitk::DataNode::New();
  coneNode->SetData(activeToolData);
  coneNode->SetName(label);
  coneNode->GetPropertyList()->SetProperty("name", mitk::StringProperty::New( label ));
  coneNode->GetPropertyList()->SetProperty("layer", mitk::IntProperty::New(0));
  coneNode->GetPropertyList()->SetProperty("visible", mitk::BoolProperty::New(true));
  coneNode->SetColor(1.0,0.0,0.0);
  coneNode->SetOpacity(0.85);
  coneNode->Modified();

  return coneNode;
}


//-----------------------------------------------------------------------------
mitk::DataNode::Pointer CreateConeRepresentation(
    const char* label,
    mitk::Vector3D& direction
    )
{
  mitk::Vector3D centerPoint;
  centerPoint[0] = 0;
  centerPoint[1] = 0;
  centerPoint[2] = 7.5;

  return CreateConeRepresentation(label, centerPoint, direction);
}

} // end namespace

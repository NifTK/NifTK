/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkSurfaceBasedRegistration.h"

namespace mitk
{

//-----------------------------------------------------------------------------
SurfaceBasedRegistration::SurfaceBasedRegistration()
{
}


//-----------------------------------------------------------------------------
SurfaceBasedRegistration::~SurfaceBasedRegistration()
{
}


//-----------------------------------------------------------------------------
void SurfaceBasedRegistration::Update(const mitk::DataNode::Pointer fixedNode,
                                      const mitk::DataNode::Pointer movingNode,
                                      vtkMatrix4x4* transformMovingToFixed)
{

}

} // end namespace


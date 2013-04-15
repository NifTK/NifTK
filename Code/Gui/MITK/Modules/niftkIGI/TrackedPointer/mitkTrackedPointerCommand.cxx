/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTrackedPointerCommand.h"

namespace mitk
{

//-----------------------------------------------------------------------------
TrackedPointerCommand::TrackedPointerCommand()
{
}


//-----------------------------------------------------------------------------
TrackedPointerCommand::~TrackedPointerCommand()
{
}


//-----------------------------------------------------------------------------
void TrackedPointerCommand::Update(const mitk::DataStorage::Pointer dataStorage,
         const mitk::DataNode::Pointer surfaceNode,
         const mitk::DataNode::Pointer pointerToWorldNode,
         const vtkMatrix4x4* tipToPointerTransform)
{

}

//-----------------------------------------------------------------------------
} // end namespace


/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkTrackedImageCommand.h"

namespace mitk
{

//-----------------------------------------------------------------------------
TrackedImageCommand::TrackedImageCommand()
{
}


//-----------------------------------------------------------------------------
TrackedImageCommand::~TrackedImageCommand()
{
}


//-----------------------------------------------------------------------------
void TrackedImageCommand::Update(const mitk::DataStorage::Pointer dataStorage,
                                 const mitk::DataNode::Pointer imageNode,
                                 const mitk::DataNode::Pointer surfaceNode,
                                 const mitk::DataNode::Pointer probeToWorldNode,
                                 const vtkMatrix4x4* imageToProbeTransform)
{

}

} // end namespace


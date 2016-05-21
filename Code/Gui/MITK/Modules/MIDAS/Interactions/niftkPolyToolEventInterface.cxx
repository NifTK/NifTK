/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPolyToolEventInterface.h"

#include "niftkPolyTool.h"

namespace niftk
{

MIDASPolyToolEventInterface::MIDASPolyToolEventInterface()
: m_Tool(NULL)
{
}

MIDASPolyToolEventInterface::~MIDASPolyToolEventInterface()
{
}

void MIDASPolyToolEventInterface::SetMIDASPolyTool( MIDASPolyTool* tool )
{
  m_Tool = tool;
}

void MIDASPolyToolEventInterface::ExecuteOperation(mitk::Operation* op)
{
  m_Tool->ExecuteOperation(op);
}

}

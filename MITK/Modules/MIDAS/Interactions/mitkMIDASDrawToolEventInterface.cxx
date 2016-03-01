/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASDrawToolEventInterface.h"
#include "mitkMIDASDrawTool.h"

namespace mitk
{

MIDASDrawToolEventInterface::MIDASDrawToolEventInterface()
: m_Tool(NULL)
{
}

MIDASDrawToolEventInterface::~MIDASDrawToolEventInterface()
{
}

void MIDASDrawToolEventInterface::SetMIDASDrawTool( MIDASDrawTool* tool )
{
  m_Tool = tool;
}

void MIDASDrawToolEventInterface::ExecuteOperation(mitk::Operation* op)
{
  m_Tool->ExecuteOperation(op);
}

} // end namespace

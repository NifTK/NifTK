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

PolyToolEventInterface::PolyToolEventInterface()
: m_Tool(NULL)
{
}

PolyToolEventInterface::~PolyToolEventInterface()
{
}

void PolyToolEventInterface::SetPolyTool( PolyTool* tool )
{
  m_Tool = tool;
}

void PolyToolEventInterface::ExecuteOperation(mitk::Operation* op)
{
  m_Tool->ExecuteOperation(op);
}

}

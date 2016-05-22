/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkDrawToolEventInterface.h"

#include "niftkDrawTool.h"

namespace niftk
{

DrawToolEventInterface::DrawToolEventInterface()
: m_Tool(NULL)
{
}

DrawToolEventInterface::~DrawToolEventInterface()
{
}

void DrawToolEventInterface::SetDrawTool( DrawTool* tool )
{
  m_Tool = tool;
}

void DrawToolEventInterface::ExecuteOperation(mitk::Operation* op)
{
  m_Tool->ExecuteOperation(op);
}

}

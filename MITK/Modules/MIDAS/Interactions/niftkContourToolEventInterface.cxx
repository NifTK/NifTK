/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkContourToolEventInterface.h"

#include "niftkContourTool.h"

namespace niftk
{

ContourToolEventInterface::ContourToolEventInterface()
: m_Tool(NULL)
{
}

ContourToolEventInterface::~ContourToolEventInterface()
{
}

void ContourToolEventInterface::SetContourTool( ContourTool* tool )
{
  m_Tool = tool;
}

void ContourToolEventInterface::ExecuteOperation(mitk::Operation* op)
{
  m_Tool->ExecuteOperation(op);
}

}

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMIDASContourToolEventInterface.h"
#include "niftkMIDASContourTool.h"

namespace niftk
{

MIDASContourToolEventInterface::MIDASContourToolEventInterface()
: m_Tool(NULL)
{
}

MIDASContourToolEventInterface::~MIDASContourToolEventInterface()
{
}

void MIDASContourToolEventInterface::SetMIDASContourTool( MIDASContourTool* tool )
{
  m_Tool = tool;
}

void MIDASContourToolEventInterface::ExecuteOperation(mitk::Operation* op)
{
  m_Tool->ExecuteOperation(op);
}

}

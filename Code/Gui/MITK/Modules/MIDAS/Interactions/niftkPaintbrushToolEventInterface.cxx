/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPaintbrushToolEventInterface.h"
#include "niftkPaintbrushTool.h"

namespace niftk
{

MIDASPaintbrushToolEventInterface::MIDASPaintbrushToolEventInterface()
: m_MIDASPaintbrushTool(NULL)
{
}

MIDASPaintbrushToolEventInterface::~MIDASPaintbrushToolEventInterface()
{
}

void MIDASPaintbrushToolEventInterface::SetMIDASPaintbrushTool( MIDASPaintbrushTool* paintbrushTool )
{
  m_MIDASPaintbrushTool = paintbrushTool;
}

void MIDASPaintbrushToolEventInterface::ExecuteOperation(mitk::Operation* op)
{
  m_MIDASPaintbrushTool->ExecuteOperation(op);
}

}

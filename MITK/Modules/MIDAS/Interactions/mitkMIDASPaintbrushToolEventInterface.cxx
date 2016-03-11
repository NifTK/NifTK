/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASPaintbrushToolEventInterface.h"
#include "mitkMIDASPaintbrushTool.h"

namespace mitk
{

MIDASPaintbrushToolEventInterface::MIDASPaintbrushToolEventInterface()
: m_MIDASPaintBrushTool(NULL)
{
}

MIDASPaintbrushToolEventInterface::~MIDASPaintbrushToolEventInterface()
{
}

void MIDASPaintbrushToolEventInterface::SetMIDASPaintbrushTool( MIDASPaintbrushTool* paintbrushTool )
{
  m_MIDASPaintBrushTool = paintbrushTool;
}

void MIDASPaintbrushToolEventInterface::ExecuteOperation(mitk::Operation* op)
{
  m_MIDASPaintBrushTool->ExecuteOperation(op);
}

} // end namespace

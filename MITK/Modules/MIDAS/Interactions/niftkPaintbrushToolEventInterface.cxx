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

PaintbrushToolEventInterface::PaintbrushToolEventInterface()
: m_PaintbrushTool(NULL)
{
}

PaintbrushToolEventInterface::~PaintbrushToolEventInterface()
{
}

void PaintbrushToolEventInterface::SetPaintbrushTool(PaintbrushTool* paintbrushTool)
{
  m_PaintbrushTool = paintbrushTool;
}

void PaintbrushToolEventInterface::ExecuteOperation(mitk::Operation* op)
{
  m_PaintbrushTool->ExecuteOperation(op);
}

}

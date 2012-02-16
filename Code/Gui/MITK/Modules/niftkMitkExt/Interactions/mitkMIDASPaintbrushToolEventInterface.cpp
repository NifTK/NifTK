/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

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

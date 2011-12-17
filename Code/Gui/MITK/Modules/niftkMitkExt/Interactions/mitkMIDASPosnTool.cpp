/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-07-19 12:16:16 +0100 (Tue, 19 Jul 2011) $
 Revision          : $Revision: 6802 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "mitkMIDASPosnTool.h"
#include "mitkMIDASPosnTool.xpm"
#include "mitkImageAccessByItk.h"
#include "mitkToolManager.h"

#include "itkImageRegionConstIterator.h"

namespace mitk{
  MITK_TOOL_MACRO(NIFTKMITKEXT_EXPORT, MIDASPosnTool, "MIDAS Posn Tool");
}

mitk::MIDASPosnTool::MIDASPosnTool() : MIDASTool("dummy")
{

}

mitk::MIDASPosnTool::~MIDASPosnTool()
{

}

const char* mitk::MIDASPosnTool::GetName() const
{
  return "Posn";
}

const char** mitk::MIDASPosnTool::GetXPM() const
{
  return mitkMIDASPosnTool_xpm;
}

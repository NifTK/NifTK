/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMIDASPosnTool.h"
#include "mitkMIDASPosnTool.xpm"
#include <mitkImageAccessByItk.h>
#include <mitkToolManager.h>

#include <itkImageRegionConstIterator.h>

namespace mitk
{
  MITK_TOOL_MACRO(NIFTKMIDAS_EXPORT, MIDASPosnTool, "MIDAS Posn Tool");
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

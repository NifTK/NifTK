/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPosnTool.h"
#include "niftkPosnTool.xpm"
#include <mitkImageAccessByItk.h>
#include <mitkToolManager.h>

#include <itkImageRegionConstIterator.h>

#include "niftkToolFactoryMacros.h"

namespace niftk
{

NIFTK_TOOL_MACRO(NIFTKMIDAS_EXPORT, MIDASPosnTool, "MIDAS Posn Tool");

MIDASPosnTool::MIDASPosnTool()
: MIDASTool()
{
}

MIDASPosnTool::~MIDASPosnTool()
{
}

const char* MIDASPosnTool::GetName() const
{
  return "Posn";
}

const char** MIDASPosnTool::GetXPM() const
{
  return niftkPosnTool_xpm;
}

}

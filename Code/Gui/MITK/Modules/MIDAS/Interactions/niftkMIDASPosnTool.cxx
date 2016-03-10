/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkMIDASPosnTool.h"
#include "niftkMIDASPosnTool.xpm"
#include <mitkImageAccessByItk.h>
#include <mitkToolManager.h>

#include <itkImageRegionConstIterator.h>

namespace niftk
{
  MITK_TOOL_MACRO(NIFTKMIDAS_EXPORT, MIDASPosnTool, "MIDAS Posn Tool");
}

niftk::MIDASPosnTool::MIDASPosnTool()
: MIDASTool()
{
}

niftk::MIDASPosnTool::~MIDASPosnTool()
{
}

const char* niftk::MIDASPosnTool::GetName() const
{
  return "Posn";
}

const char** niftk::MIDASPosnTool::GetXPM() const
{
  return niftkMIDASPosnTool_xpm;
}

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkPosnTool.h"

#include <itkImageRegionConstIterator.h>

#include <mitkImageAccessByItk.h>
#include <mitkToolManager.h>

#include "niftkPosnTool.xpm"
#include "niftkToolFactoryMacros.h"

namespace niftk
{

NIFTK_TOOL_MACRO(NIFTKMIDAS_EXPORT, PosnTool, "Posn Tool");

PosnTool::PosnTool()
: Tool()
{
}

PosnTool::~PosnTool()
{
}

const char* PosnTool::GetName() const
{
  return "Posn";
}

const char** PosnTool::GetXPM() const
{
  return niftkPosnTool_xpm;
}

}

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkNiftyMIDASAppWorkbenchAdvisor.h"

//-----------------------------------------------------------------------------
std::string QmitkNiftyMIDASAppWorkbenchAdvisor::GetInitialWindowPerspectiveId()
{
  return "uk.ac.ucl.cmic.gui.qt.niftyview.midasperspective";
}

//-----------------------------------------------------------------------------
std::string QmitkNiftyMIDASAppWorkbenchAdvisor::GetWindowIconResourcePath() const
{
  return ":/QmitkNiftyMIDASApplication/icon_ion.xpm";
}

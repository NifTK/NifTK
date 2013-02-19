/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkNiftyViewAppWorkbenchAdvisor.h"

//-----------------------------------------------------------------------------
std::string QmitkNiftyViewAppWorkbenchAdvisor::GetInitialWindowPerspectiveId()
{
  return "uk.ac.ucl.cmic.gui.qt.niftyview.midasperspective";
}

//-----------------------------------------------------------------------------
std::string QmitkNiftyViewAppWorkbenchAdvisor::GetWindowIconResourcePath() const
{
  return ":/QmitkNiftyViewApplication/icon_ucl.xpm";
}

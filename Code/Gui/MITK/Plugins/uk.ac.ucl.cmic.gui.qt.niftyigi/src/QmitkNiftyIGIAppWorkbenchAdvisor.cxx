/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkNiftyIGIAppWorkbenchAdvisor.h"
#include "QmitkNiftyIGIWorkbenchWindowAdvisor.h"

//-----------------------------------------------------------------------------
std::string QmitkNiftyIGIAppWorkbenchAdvisor::GetInitialWindowPerspectiveId()
{
  return "uk.ac.ucl.cmic.gui.qt.niftyview.igiperspective";
}

//-----------------------------------------------------------------------------
std::string QmitkNiftyIGIAppWorkbenchAdvisor::GetWindowIconResourcePath() const
{
  return ":/QmitkNiftyIGIApplication/icon_cmic.xpm";
}

//-----------------------------------------------------------------------------
QmitkBaseWorkbenchWindowAdvisor* QmitkNiftyIGIAppWorkbenchAdvisor::CreateQmitkBaseWorkbenchWindowAdvisor(
    berry::IWorkbenchWindowConfigurer::Pointer configurer)
{
  return new QmitkNiftyIGIWorkbenchWindowAdvisor(this, configurer);
}

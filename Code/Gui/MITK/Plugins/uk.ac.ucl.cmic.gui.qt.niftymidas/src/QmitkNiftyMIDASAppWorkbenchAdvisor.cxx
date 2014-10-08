/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkNiftyMIDASAppWorkbenchAdvisor.h"
#include "QmitkNiftyMIDASWorkbenchWindowAdvisor.h"

#include <berryIWorkbenchConfigurer.h>

#include <mitkLogMacros.h>


//-----------------------------------------------------------------------------
std::string QmitkNiftyMIDASAppWorkbenchAdvisor::GetInitialWindowPerspectiveId()
{
  return "uk.ac.ucl.cmic.gui.qt.niftymidas.segmentation_perspective";
}


//-----------------------------------------------------------------------------
std::string QmitkNiftyMIDASAppWorkbenchAdvisor::GetWindowIconResourcePath() const
{
  return ":/QmitkNiftyMIDASApplication/icon_ion.xpm";
}


//-----------------------------------------------------------------------------
QmitkBaseWorkbenchWindowAdvisor* QmitkNiftyMIDASAppWorkbenchAdvisor::CreateQmitkBaseWorkbenchWindowAdvisor(
    berry::IWorkbenchWindowConfigurer::Pointer configurer)
{
  return new QmitkNiftyMIDASWorkbenchWindowAdvisor(this, configurer);
}


//-----------------------------------------------------------------------------
void QmitkNiftyMIDASAppWorkbenchAdvisor::PostStartup()
{
  std::vector<std::string> args = berry::Platform::GetApplicationArgs();
  for (std::vector<std::string>::const_iterator it = args.begin(); it != args.end(); ++it)
  {
    std::string arg = *it;
    if (arg == "--perspective")
    {
      ++it;
      if (it == args.end())
      {
        break;
      }

      std::string perspectiveLabel = *it;

      berry::IWorkbenchConfigurer::Pointer workbenchConfigurer = this->GetWorkbenchConfigurer();
      berry::IWorkbench* workbench = workbenchConfigurer->GetWorkbench();
      berry::IPerspectiveRegistry* perspectiveRegistry = workbench->GetPerspectiveRegistry();
      berry::IPerspectiveDescriptor::Pointer perspectiveDescriptor = perspectiveRegistry->FindPerspectiveWithLabel(perspectiveLabel);

      if (perspectiveDescriptor.IsNull())
      {
        MITK_ERROR << "Unknown perspective.";
        continue;
      }

      std::vector<berry::IWorkbenchWindow::Pointer> workbenchWindows = workbench->GetWorkbenchWindows();
      for (std::vector<berry::IWorkbenchWindow::Pointer>::iterator workbenchWindowIt = workbenchWindows.begin();
           workbenchWindowIt != workbenchWindows.end();
           ++workbenchWindowIt)
      {
        workbench->ShowPerspective(perspectiveDescriptor->GetId(), *workbenchWindowIt);
      }
    }
  }
}

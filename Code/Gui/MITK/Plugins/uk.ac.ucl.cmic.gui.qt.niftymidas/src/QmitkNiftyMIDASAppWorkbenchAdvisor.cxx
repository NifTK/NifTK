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
#include <QmitkMultiViewerEditor.h>
#include <niftkMultiViewerWidget.h>

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

  berry::IWorkbenchConfigurer::Pointer workbenchConfigurer = this->GetWorkbenchConfigurer();
  berry::IWorkbench* workbench = workbenchConfigurer->GetWorkbench();
  berry::IWorkbenchWindow::Pointer activeWorkbenchWindow = workbench->GetActiveWorkbenchWindow();

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

      berry::IPerspectiveRegistry* perspectiveRegistry = workbench->GetPerspectiveRegistry();
      berry::IPerspectiveDescriptor::Pointer perspectiveDescriptor = perspectiveRegistry->FindPerspectiveWithLabel(perspectiveLabel);

      if (perspectiveDescriptor.IsNull())
      {
        MITK_ERROR << "Unknown perspective.";
        continue;
      }

      workbench->ShowPerspective(perspectiveDescriptor->GetId(), activeWorkbenchWindow);
    }
    else if (arg == "--window-layout")
    {
      ++it;
      if (it == args.end())
      {
        break;
      }

      std::string windowLayoutName = *it;

      WindowLayout windowLayout = ::GetWindowLayout(windowLayoutName);

      if (windowLayout != WINDOW_LAYOUT_UNKNOWN)
      {
        berry::IEditorPart::Pointer activeEditor = activeWorkbenchWindow->GetActivePage()->GetActiveEditor();
        QmitkMultiViewerEditor* dndDisplay = dynamic_cast<QmitkMultiViewerEditor*>(activeEditor.GetPointer());
        niftkMultiViewerWidget* multiViewer = dndDisplay->GetMultiViewer();
        multiViewer->SetDefaultWindowLayout(windowLayout);
      }
    }
  }
}

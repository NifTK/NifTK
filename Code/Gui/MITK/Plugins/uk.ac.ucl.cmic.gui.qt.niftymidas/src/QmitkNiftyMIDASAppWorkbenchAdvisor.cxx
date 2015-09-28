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
#include <berryPlatform.h>

#include <QmitkCommonAppsApplicationPlugin.h>

//-----------------------------------------------------------------------------
QString QmitkNiftyMIDASAppWorkbenchAdvisor::GetInitialWindowPerspectiveId()
{
  return "uk.ac.ucl.cmic.gui.qt.niftymidas.segmentation_perspective";
}


//-----------------------------------------------------------------------------
QString QmitkNiftyMIDASAppWorkbenchAdvisor::GetWindowIconResourcePath() const
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
  Superclass::PostStartup();

  QStringList args = berry::Platform::GetApplicationArgs();

  berry::IWorkbenchConfigurer::Pointer workbenchConfigurer = this->GetWorkbenchConfigurer();
  berry::IWorkbench* workbench = workbenchConfigurer->GetWorkbench();
  berry::IWorkbenchWindow::Pointer workbenchWindow = workbench->GetActiveWorkbenchWindow();
  if (!workbenchWindow)
  {
    QList<berry::IWorkbenchWindow::Pointer> workbenchWindows = workbench->GetWorkbenchWindows();
    if (!workbenchWindows.empty())
    {
      workbenchWindow = workbenchWindows[0];
    }
    else
    {
      /// TODO there is no active workbench window.
      MITK_ERROR << "There is no active workbench window.";
    }
  }

  for (QStringList::const_iterator it = args.begin(); it != args.end(); ++it)
  {
    QString arg = *it;
    if (arg == QString("--perspective"))
    {
      if (it + 1 == args.end()
          || (it + 1)->isEmpty()
          || (*(it + 1))[0] == '-')
      {
        MITK_ERROR << "Invalid arguments: perspective name missing.";
        continue;
      }

      ++it;
      QString perspectiveLabel = *it;

      berry::IPerspectiveRegistry* perspectiveRegistry = workbench->GetPerspectiveRegistry();
      berry::IPerspectiveDescriptor::Pointer perspectiveDescriptor = perspectiveRegistry->FindPerspectiveWithLabel(perspectiveLabel);

      if (perspectiveDescriptor.IsNull())
      {
        MITK_ERROR << "Invalid arguments: unknown perspective.";
        continue;
      }

      workbench->ShowPerspective(perspectiveDescriptor->GetId(), workbenchWindow);
    }
  }
}

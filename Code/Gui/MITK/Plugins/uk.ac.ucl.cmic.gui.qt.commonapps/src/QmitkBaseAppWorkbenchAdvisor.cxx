/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkBaseAppWorkbenchAdvisor.h"
#include "QmitkBaseWorkbenchWindowAdvisor.h"

//-----------------------------------------------------------------------------
void QmitkBaseAppWorkbenchAdvisor::Initialize(berry::IWorkbenchConfigurer::Pointer configurer)
{
  berry::QtWorkbenchAdvisor::Initialize(configurer);
  configurer->SetSaveAndRestore(true);
}


//-----------------------------------------------------------------------------
berry::WorkbenchWindowAdvisor* QmitkBaseAppWorkbenchAdvisor::CreateWorkbenchWindowAdvisor(
        berry::IWorkbenchWindowConfigurer::Pointer configurer)
{
  // Create the advisor, or derived classes can create their own.
  QmitkBaseWorkbenchWindowAdvisor* advisor = this->CreateQmitkBaseWorkbenchWindowAdvisor(configurer);

  // Exclude the help perspective from org.blueberry.ui.qt.help from the normal perspective list.
  // The perspective gets a dedicated menu entry in the help menu.

  std::vector<std::string> excludePerspectives;
  excludePerspectives.push_back("org.blueberry.perspectives.help");

  advisor->SetPerspectiveExcludeList(excludePerspectives);
  advisor->SetWindowIcon(this->GetWindowIconResourcePath());

  return advisor;
}


//-----------------------------------------------------------------------------
QmitkBaseWorkbenchWindowAdvisor* QmitkBaseAppWorkbenchAdvisor::CreateQmitkBaseWorkbenchWindowAdvisor(
    berry::IWorkbenchWindowConfigurer::Pointer configurer)
{
  return new QmitkBaseWorkbenchWindowAdvisor(this, configurer);
}

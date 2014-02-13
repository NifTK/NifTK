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
#include <QMessageBox>

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


//-----------------------------------------------------------------------------
bool QmitkBaseAppWorkbenchAdvisor::PreShutdown()
{
  QMessageBox msgBox;
  // would be nice if we could include application's name here.
  msgBox.setText("Do you want to close the application?");
  msgBox.setInformativeText("Make sure you have saved everything you deem important!");
  msgBox.setIcon(QMessageBox::Information);
  msgBox.setStandardButtons(QMessageBox::Yes | QMessageBox::No);
  msgBox.setDefaultButton(QMessageBox::No);
  int r = msgBox.exec();

  // only close the application if user really clicked on "yes".
  // everything else will simply cancel the dialog box and keep us running.
  bool  okToClose = r == QMessageBox::Yes;
  // ask base class as well.
  okToClose &= QtWorkbenchAdvisor::PreShutdown();

  return okToClose;
}

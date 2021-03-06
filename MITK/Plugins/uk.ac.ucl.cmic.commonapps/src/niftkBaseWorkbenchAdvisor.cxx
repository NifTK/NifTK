/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseWorkbenchAdvisor.h"

#include <QMessageBox>

#include <mitkDataStorage.h>
#include <mitkLogMacros.h>

#include "niftkBaseWorkbenchWindowAdvisor.h"
#include "internal/niftkPluginActivator.h"


namespace niftk
{

//-----------------------------------------------------------------------------
void BaseWorkbenchAdvisor::Initialize(berry::IWorkbenchConfigurer::Pointer configurer)
{
  berry::QtWorkbenchAdvisor::Initialize(configurer);
  configurer->SetSaveAndRestore(true);
}


//-----------------------------------------------------------------------------
berry::WorkbenchWindowAdvisor* BaseWorkbenchAdvisor::CreateWorkbenchWindowAdvisor(
        berry::IWorkbenchWindowConfigurer::Pointer configurer)
{
  // Create the advisor, or derived classes can create their own.
  BaseWorkbenchWindowAdvisor* advisor = this->CreateBaseWorkbenchWindowAdvisor(configurer);

  // Exclude the help perspective from org.blueberry.ui.qt.help from the normal perspective list.
  // The perspective gets a dedicated menu entry in the help menu.

  QStringList excludePerspectives;
  excludePerspectives.push_back("org.blueberry.perspectives.help");

  advisor->SetPerspectiveExcludeList(excludePerspectives);
  advisor->SetWindowIcon(this->GetWindowIconResourcePath());

  return advisor;
}


//-----------------------------------------------------------------------------
BaseWorkbenchWindowAdvisor* BaseWorkbenchAdvisor::CreateBaseWorkbenchWindowAdvisor(
    berry::IWorkbenchWindowConfigurer::Pointer configurer)
{
  return new BaseWorkbenchWindowAdvisor(this, configurer);
}


// --------------------------------------------------------------------------
mitk::DataStorage* BaseWorkbenchAdvisor::GetDataStorage()
{
  mitk::DataStorage::Pointer dataStorage = 0;

  ctkPluginContext* context = PluginActivator::GetInstance()->GetContext();
  ctkServiceReference dsServiceRef = context->getServiceReference<mitk::IDataStorageService>();
  if (dsServiceRef)
  {
    mitk::IDataStorageService* dsService = context->getService<mitk::IDataStorageService>(dsServiceRef);
    if (dsService)
    {
      mitk::IDataStorageReference::Pointer dataStorageRef = dsService->GetActiveDataStorage();
      dataStorage = dataStorageRef->GetDataStorage();
    }
  }

  return dataStorage;
}


//-----------------------------------------------------------------------------
void BaseWorkbenchAdvisor::PostStartup()
{
  QString perspectiveLabel = PluginActivator::GetInstance()->GetContext()->getProperty("applicationArgs.perspective").toString();

  if (!perspectiveLabel.isEmpty())
  {
    this->SetPerspective(perspectiveLabel);
  }

  QVariant resetArg = PluginActivator::GetInstance()->GetContext()->getProperty("applicationArgs.reset-perspective");

  if (resetArg.isValid())
  {
    this->ResetPerspective();
  }
}


//-----------------------------------------------------------------------------
void BaseWorkbenchAdvisor::SetPerspective(const QString& perspectiveLabel)
{
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
      MITK_ERROR << "There is no active workbench window. Cannot set perspective.";
      return;
    }
  }

  berry::IPerspectiveRegistry* perspectiveRegistry = workbench->GetPerspectiveRegistry();
  berry::IPerspectiveDescriptor::Pointer perspectiveDescriptor = perspectiveRegistry->FindPerspectiveWithLabel(perspectiveLabel);

  if (perspectiveDescriptor.IsNull())
  {
    MITK_ERROR << "Invalid arguments: unknown perspective.";
    return;
  }

  workbench->ShowPerspective(perspectiveDescriptor->GetId(), workbenchWindow);
}


//-----------------------------------------------------------------------------
void BaseWorkbenchAdvisor::ResetPerspective()
{
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
      MITK_ERROR << "There is no active workbench window. Cannot reset perspective.";
      return;
    }
  }

  workbenchWindow->GetActivePage()->ResetPerspective();
}


//-----------------------------------------------------------------------------
bool BaseWorkbenchAdvisor::PreShutdown()
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

}

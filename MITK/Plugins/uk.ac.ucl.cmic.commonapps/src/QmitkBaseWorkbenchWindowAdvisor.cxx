/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QmitkBaseWorkbenchWindowAdvisor.h"

#include <cstring>

#include "QmitkCommonAppsApplicationPlugin.h"
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QAction>
#include <QList>
#include <QApplication>
#include <niftkHelpAboutDialog.h>
#include <mitkDataNode.h>
#include <mitkDataStorage.h>
#include <mitkDataStorageEditorInput.h>
#include <mitkIDataStorageService.h>
#include <mitkNodePredicateData.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateProperty.h>
#include <mitkProperties.h>
#include <mitkRenderingManager.h>
#include <NifTKConfigure.h>

//-----------------------------------------------------------------------------
QmitkBaseWorkbenchWindowAdvisor::QmitkBaseWorkbenchWindowAdvisor(
    berry::WorkbenchAdvisor* wbAdvisor,
    berry::IWorkbenchWindowConfigurer::Pointer configurer)
: QmitkExtWorkbenchWindowAdvisor(wbAdvisor, configurer)
{
}


//-----------------------------------------------------------------------------
void QmitkBaseWorkbenchWindowAdvisor::OnHelpAbout()
{
  niftk::HelpAboutDialog *dialog = new niftk::HelpAboutDialog(QApplication::activeWindow(), QApplication::applicationName());
  dialog->setModal(true);
  dialog->show();
}


//-----------------------------------------------------------------------------
void QmitkBaseWorkbenchWindowAdvisor::PreWindowOpen()
{
  this->ShowMitkVersionInfo(false); // Please look in niftk::HelpAboutDialog.h
  this->ShowVersionInfo(false);     // Please look in niftk::HelpAboutDialog.h

  QmitkExtWorkbenchWindowAdvisor::PreWindowOpen();

  // When the GUI starts, these views are not shown.
  QStringList viewExcludeList = this->GetViewExcludeList();
  viewExcludeList.push_back("org.mitk.views.modules");
  viewExcludeList.push_back("org.blueberry.views.helpcontents");
  viewExcludeList.push_back("org.blueberry.views.helpindex");
  viewExcludeList.push_back("org.blueberry.views.helpsearch");
  viewExcludeList.push_back("org.mitk.views.imagestatistics");
  this->SetViewExcludeList(viewExcludeList);
}


//-----------------------------------------------------------------------------
void QmitkBaseWorkbenchWindowAdvisor::PostWindowCreate()
{
  QmitkExtWorkbenchWindowAdvisor::PostWindowCreate();

  // Get rid of Welcome menu item, and re-connect the About menu item.
  //
  // 1. Get hold of menu bar
  berry::IWorkbenchWindow::Pointer window =
   this->GetWindowConfigurer()->GetWindow();
  QMainWindow* mainWindow =
   static_cast<QMainWindow*> (window->GetShell()->GetControl());
  QMenuBar* menuBar = mainWindow->menuBar();
  QList<QMenu *> allMenus = menuBar->findChildren<QMenu *>();

  for (int i = 0; i < allMenus.count(); i++)
  {
    QList<QAction*> actionsForMenu = allMenus.at(i)->findChildren<QAction*>();
    for (int j = 0; j < actionsForMenu.count(); j++)
    {
      QAction *action = actionsForMenu.at(j);

      if (action != NULL && action->text() == "&About")
      {
        // 1.1. Disconnect existing slot
        action->disconnect();

        // 1.2. Reconnect slot to our method to call our About Dialog.
        QObject::connect(action, SIGNAL(triggered()), this, SLOT(OnHelpAbout()));
      }

      if (action != NULL && action->text() == "&Welcome")
      {
        action->setVisible(false);
      }
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkBaseWorkbenchWindowAdvisor::OpenEditor(const QString& editorName)
{
  berry::IWorkbenchWindow::Pointer wnd = this->GetWindowConfigurer()->GetWindow();
  berry::IWorkbenchPage::Pointer page = wnd->GetActivePage();
  ctkPluginContext* context = QmitkCommonAppsApplicationPlugin::GetDefault()->GetPluginContext();
  ctkServiceReference dsServiceRef = context->getServiceReference<mitk::IDataStorageService>();
  if (dsServiceRef)
  {
    mitk::IDataStorageService* dsService = context->getService<mitk::IDataStorageService>(dsServiceRef);
    if (dsService)
    {
      berry::IEditorInput::Pointer dsInput(new mitk::DataStorageEditorInput(dsService->GetActiveDataStorage()));
      page->OpenEditor(dsInput, editorName, false, berry::IWorkbenchPage::MATCH_ID);
    }
  }
}


//-----------------------------------------------------------------------------
void QmitkBaseWorkbenchWindowAdvisor::OpenEditorIfEnvironmentVariableIsON(const std::string& envVariable, const QString& editorName)
{
  const char* envVarValue = std::getenv(envVariable.c_str());
  if (envVarValue && std::strcmp(envVarValue, "ON") == 0)
  {
    this->OpenEditor(editorName);
  }
}

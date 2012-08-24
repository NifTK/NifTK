/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-20 14:25:53 +0000 (Sun, 20 Nov 2011) $
 Revision          : $Revision: 7818 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkNiftyViewWorkbenchWindowAdvisor.h"
#include "internal/QmitkNiftyViewApplicationPlugin.h"
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>
#include <QAction>
#include <QList>
#include <QApplication>
#include "QmitkHelpAboutDialog.h"
#include "mitkDataNode.h"
#include "mitkDataNodeFactory.h"
#include "mitkDataStorage.h"
#include "mitkDataStorageEditorInput.h"
#include "mitkIDataStorageService.h"
#include "mitkNodePredicateData.h"
#include "mitkNodePredicateNot.h"
#include "mitkNodePredicateProperty.h"
#include "mitkProperties.h"
#include "mitkRenderingManager.h"
#include "EnvironmentHelper.h"
#include "NifTKConfigure.h"

QmitkNiftyViewWorkbenchWindowAdvisor::QmitkNiftyViewWorkbenchWindowAdvisor(
    berry::WorkbenchAdvisor* wbAdvisor,
    berry::IWorkbenchWindowConfigurer::Pointer configurer)
: QmitkExtWorkbenchWindowAdvisor(wbAdvisor, configurer)
{

}

void QmitkNiftyViewWorkbenchWindowAdvisor::OnHelpAbout()
{
  QmitkHelpAboutDialog *dialog = new QmitkHelpAboutDialog(QApplication::activeWindow(), QApplication::applicationName());
  dialog->setModal(true);
  dialog->show();
}

void QmitkNiftyViewWorkbenchWindowAdvisor::PreWindowOpen()
{
  this->ShowMitkVersionInfo(false); // Please look in QmitkHelpAboutDialog.h
  this->ShowVersionInfo(false);     // Please look in QmitkHelpAboutDialog.h

  QmitkExtWorkbenchWindowAdvisor::PreWindowOpen();

  // When the GUI starts, I don't want the Modules plugin to be visible.
  std::vector<std::string> viewExcludeList = this->GetViewExcludeList();
  viewExcludeList.push_back("org.mitk.views.modules");
  viewExcludeList.push_back("org.blueberry.views.helpcontents");
  viewExcludeList.push_back("org.blueberry.views.helpindex");
  viewExcludeList.push_back("org.blueberry.views.helpsearch");
  this->SetViewExcludeList(viewExcludeList);
}

void QmitkNiftyViewWorkbenchWindowAdvisor::PostWindowCreate()
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

  // In NiftyView, I have set in the midaseditor plugin.xml for the Midas Drag and Drop editor to be default.
  // This section is to try and force the standard MITK Display editor open.
  // It is assumed to be off, unless the user sets the NIFTK_MITK_DISPLAY=ON.
  // It is imagined that at the DRC, they will not want this viewer as much.
  // The unfortunate side effect is that the GUI will not correctly remember which is the ordering of the editors.
  // So, restoring a project from a project file may have the wrong viewer on-top on start-up.
  if (niftk::BooleanEnvironmentVariableIsOn("NIFTK_MITK_DISPLAY"))
  {
    berry::IWorkbenchWindow::Pointer wnd = this->GetWindowConfigurer()->GetWindow();
    berry::IWorkbenchPage::Pointer page = wnd->GetActivePage();
    ctkPluginContext* context = QmitkNiftyViewApplicationPlugin::GetDefault()->GetPluginContext();
    ctkServiceReference dsServiceRef = context->getServiceReference<mitk::IDataStorageService>();
    if (dsServiceRef)
    {
      mitk::IDataStorageService* dsService = context->getService<mitk::IDataStorageService>(dsServiceRef);
      if (dsService)
      {
        berry::IEditorInput::Pointer dsInput(new mitk::DataStorageEditorInput(dsService->GetActiveDataStorage()));
        // Use MATCH_ID as matching strategy, otherwise another editor using the same input
        // might by reused but we explicitly want an editor instance with id org.mitk.editors.stdmultiwidget.
        page->OpenEditor(dsInput, "org.mitk.editors.stdmultiwidget", false, berry::IWorkbenchPage::MATCH_ID);
      }
    }
  }
}

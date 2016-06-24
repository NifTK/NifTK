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

#include <mitkDataStorage.h>

#include "QmitkCommonAppsApplicationPlugin.h"

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

  QStringList excludePerspectives;
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


// --------------------------------------------------------------------------
mitk::DataStorage* QmitkBaseAppWorkbenchAdvisor::GetDataStorage()
{
  mitk::DataStorage::Pointer dataStorage = 0;

  ctkPluginContext* context = QmitkCommonAppsApplicationPlugin::GetDefault()->GetPluginContext();
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
void QmitkBaseAppWorkbenchAdvisor::PostStartup()
{
  /// Note:
  /// The fixedLayer property is set to true for all the images opened from the command line, so that
  /// the order of the images is not overwritten by the DataManager when it opens everything from the
  /// data storage.
  /// Now that the workbench is up and the data manager plugin loaded everything, we can clear the
  /// fixedLayer property, so that the user can rearrange the layers by drag and drop in the data manager.

  mitk::DataStorage::Pointer dataStorage = this->GetDataStorage();
  mitk::DataStorage::SetOfObjects::ConstPointer nodes = dataStorage->GetAll();

  for (mitk::DataStorage::SetOfObjects::ConstIterator it = nodes->Begin(); it != nodes->End(); ++it)
  {
    mitk::DataNode* node = it->Value();
    bool fixedLayer = false;
    node->GetBoolProperty("fixedLayer", fixedLayer);
    if (fixedLayer)
    {
      node->SetBoolProperty("fixedLayer", false);
    }
  }
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

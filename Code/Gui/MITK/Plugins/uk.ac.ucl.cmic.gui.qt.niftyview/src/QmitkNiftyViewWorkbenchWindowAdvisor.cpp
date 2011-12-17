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
#include "mitkIDataStorageService.h"
#include "mitkNodePredicateData.h"
#include "mitkNodePredicateNot.h"
#include "mitkNodePredicateProperty.h"
#include "mitkProperties.h"
#include "mitkRenderingManager.h"

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
}

void QmitkNiftyViewWorkbenchWindowAdvisor::PostWindowCreate()
{
  QmitkExtWorkbenchWindowAdvisor::PostWindowCreate();

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

  // 2. Load any command line args that look like images
  QApplication *application = dynamic_cast<QApplication*>(qApp);
  if (application != NULL)
  {
    QStringList arguments = application->arguments();
    unsigned int argumentsAdded = 0;

    if (arguments.size() > 1)
    {

      mitk::IDataStorageService::Pointer service =
        berry::Platform::GetServiceRegistry().GetServiceById<mitk::IDataStorageService>(mitk::IDataStorageService::ID);

      mitk::DataStorage::Pointer dataStorage = service->GetDefaultDataStorage()->GetDataStorage();

      for (int i = 1; i < arguments.size(); i++)
      {
        QString argument = arguments[i];
        mitk::DataNodeFactory::Pointer nodeReader = mitk::DataNodeFactory::New();
        try
        {
          nodeReader->SetFileName(argument.toLocal8Bit().constData());
          nodeReader->Update();
          for ( unsigned int i = 0 ; i < nodeReader->GetNumberOfOutputs( ); ++i )
          {
            mitk::DataNode::Pointer node;
            node = nodeReader->GetOutput(i);
            if ( node->GetData() != NULL )
            {
              dataStorage->Add(node);
              argumentsAdded++;
              MITK_INFO << "QmitkExtWorkbenchWindowAdvisor::PostWindowCreate, loaded:" << argument.toLocal8Bit().constData() << std::endl;
            }
          }
        }
        catch(...)
        {
          MITK_DEBUG << "QmitkExtWorkbenchWindowAdvisor::PostWindowCreate failed to load argument:" << argument.toLocal8Bit().constData() << std::endl;
        }
      } // end for each command line argument

      if (argumentsAdded > 0)
      {
        // Get bounds of every dataset that doesn't have "includeInBoundingBox" set to false.
        mitk::NodePredicateNot::Pointer pred
          = mitk::NodePredicateNot::New(mitk::NodePredicateProperty::New("includeInBoundingBox"
          , mitk::BoolProperty::New(false)));

        mitk::DataStorage::SetOfObjects::ConstPointer rs = dataStorage->GetSubset(pred);
        mitk::TimeSlicedGeometry::Pointer bounds = dataStorage->ComputeBoundingGeometry3D(rs);
        mitk::RenderingManager::GetInstance()->InitializeViews(bounds);
      }
    }
  }
}

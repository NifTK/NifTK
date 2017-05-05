/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkBaseWorkbenchWindowAdvisor.h"

#include <cstring>

#include <QAction>
#include <QApplication>
#include <QList>
#include <QMainWindow>
#include <QMenu>
#include <QMenuBar>

#include <berryCommandContributionItemParameter.h>
#include <berryIWorkbenchCommandConstants.h>
#include <berryIWorkbenchWindowConfigurer.h>
#include <internal/berryQtOpenPerspectiveAction.h>

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
#include <niftkHelpAboutDialog.h>

#include "internal/niftkPluginActivator.h"


namespace niftk
{

class PerspectiveListener: public berry::IPerspectiveListener
{
public:

  PerspectiveListener(berry::IWorkbenchWindow::Pointer window)
    : m_Window(window),
      m_PerspectiveMenu(nullptr)
  {
    berry::IPerspectiveRegistry* perspectiveRegistry = m_Window->GetWorkbench()->GetPerspectiveRegistry();
    berry::IPerspectiveDescriptor::Pointer activePerspective = m_Window->GetActivePage()->GetPerspective();

    QMainWindow* mainWindow = static_cast<QMainWindow*>(m_Window->GetShell()->GetControl());
    QMenuBar* menuBar = mainWindow->menuBar();
    QList<QMenu*> menus = menuBar->findChildren<QMenu*>();
    for (QMenu* menu: menus)
    {
      for (QAction* action: menu->actions())
      {
        if (action->text() == "&Open Perspective")
        {
          m_PerspectiveMenu = action->menu();
          for (QAction* perspectiveAction: m_PerspectiveMenu->actions())
          {
            berry::IPerspectiveDescriptor::Pointer perspective =
                perspectiveRegistry->FindPerspectiveWithLabel(perspectiveAction->text());
            if (perspective.IsNotNull())
            {
              m_PerspectiveActions[perspective->GetId()] = perspectiveAction;
              perspectiveAction->setChecked(perspective == activePerspective);
            }
          }
          return;
        }
      }
    }
  }

  Events::Types GetPerspectiveEventTypes() const override
  {
    return Events::ACTIVATED | Events::DEACTIVATED | Events::SAVED_AS;
  }

  void PerspectiveActivated(const berry::IWorkbenchPage::Pointer& /*page*/,
    const berry::IPerspectiveDescriptor::Pointer& perspective) override
  {
    if (QAction* action = m_PerspectiveActions[perspective->GetId()])
    {
      action->setChecked(true);
    }
  }

  void PerspectiveDeactivated(const berry::IWorkbenchPage::Pointer& /*page*/,
    const berry::IPerspectiveDescriptor::Pointer& perspective) override
  {
    if (QAction* action = m_PerspectiveActions[perspective->GetId()])
    {
      action->setChecked(false);
    }
  }

  void PerspectiveSavedAs(const berry::IWorkbenchPage::Pointer& /*page*/,
                          const berry::IPerspectiveDescriptor::Pointer& /*oldPerspective*/,
                          const berry::IPerspectiveDescriptor::Pointer& newPerspective) override
  {
    QAction* firstPerspectiveAction = m_PerspectiveMenu->actions().first();
    assert(firstPerspectiveAction != nullptr);
    QActionGroup* perspectiveGroup = firstPerspectiveAction->actionGroup();

    QAction* newPerspectiveAction = new berry::QtOpenPerspectiveAction(m_Window, newPerspective, perspectiveGroup);
    newPerspectiveAction->setChecked(true);
    m_PerspectiveActions.insert(newPerspective->GetId(), newPerspectiveAction);
    bool inserted = false;
    for (QAction* perspectiveAction: m_PerspectiveMenu->actions())
    {
      if (perspectiveAction->text() > newPerspectiveAction->text())
      {
        m_PerspectiveMenu->insertAction(perspectiveAction, newPerspectiveAction);
        inserted = true;
        break;
      }
    }
    if (!inserted)
    {
      m_PerspectiveMenu->addAction(newPerspectiveAction);
    }
  }

  void PerspectiveDeleted(const berry::IPerspectiveDescriptor::Pointer& perspective)
  {
    if (QAction* action = m_PerspectiveActions[perspective->GetId()])
    {
      m_PerspectiveMenu->removeAction(action);
      m_PerspectiveActions.remove(perspective->GetId());
    }
  }

private:
  berry::IWorkbenchWindow::Pointer m_Window;
  QMenu* m_PerspectiveMenu;
  // maps perspective ids to QAction objects
  QHash<QString, QAction*> m_PerspectiveActions;
};


//-----------------------------------------------------------------------------
BaseWorkbenchWindowAdvisor::BaseWorkbenchWindowAdvisor(
    berry::WorkbenchAdvisor* wbAdvisor,
    berry::IWorkbenchWindowConfigurer::Pointer configurer)
: QmitkExtWorkbenchWindowAdvisor(wbAdvisor, configurer)
{
}


//-----------------------------------------------------------------------------
void BaseWorkbenchWindowAdvisor::OnHelpAbout()
{
  niftk::HelpAboutDialog *dialog = new niftk::HelpAboutDialog(QApplication::activeWindow(), QApplication::applicationName());
  dialog->setModal(true);
  dialog->show();
}


//-----------------------------------------------------------------------------
void BaseWorkbenchWindowAdvisor::OnDeletePerspective()
{
  berry::IWorkbenchWindowConfigurer::Pointer windowConfigurer = this->GetWindowConfigurer();
  berry::IWorkbenchWindow::Pointer workbenchWindow = windowConfigurer->GetWindow();

  berry::IWorkbench* workbench = workbenchWindow->GetWorkbench();
  berry::IPerspectiveRegistry* perspectiveRegistry = workbench->GetPerspectiveRegistry();

  berry::IWorkbenchPage::Pointer workbenchPage = workbenchWindow->GetActivePage();

  berry::IPerspectiveDescriptor::Pointer currentPerspective = workbenchPage->GetPerspective();
  berry::IPerspectiveDescriptor::Pointer defaultPerspective =
      perspectiveRegistry->FindPerspectiveWithId(perspectiveRegistry->GetDefaultPerspective());

  if (currentPerspective.IsNotNull()
      && defaultPerspective.IsNotNull()
      && currentPerspective != defaultPerspective)
  {
    workbenchPage->SetPerspective(defaultPerspective);
    perspectiveRegistry->DeletePerspective(currentPerspective);

    auto perspectiveListener = dynamic_cast<PerspectiveListener*>(m_PerspectiveListener.data());
    perspectiveListener->PerspectiveDeleted(currentPerspective);
  }
}


//-----------------------------------------------------------------------------
void BaseWorkbenchWindowAdvisor::PreWindowOpen()
{
  QString productName = PluginActivator::GetInstance()->GetContext()->getProperty("applicationArgs.product-name").toString();

  if (!productName.isEmpty())
  {
    this->SetProductName(productName);
  }

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
void BaseWorkbenchWindowAdvisor::PostWindowCreate()
{
  QmitkExtWorkbenchWindowAdvisor::PostWindowCreate();

  // Get rid of Welcome menu item, and re-connect the About menu item.
  //
  // 1. Get hold of menu bar
  berry::IWorkbenchWindow::Pointer window = this->GetWindowConfigurer()->GetWindow();
  QMainWindow* mainWindow = static_cast<QMainWindow*>(window->GetShell()->GetControl());
  QMenuBar* menuBar = mainWindow->menuBar();
  QList<QMenu *> menus = menuBar->findChildren<QMenu*>();

  for (QMenu* menu: menus)
  {
    for (QAction* action: menu->actions())
    {
      if (action->text() == "&About")
      {
        // 1.1. Disconnect existing slot
        action->disconnect();

        // 1.2. Reconnect slot to our method to call our About Dialog.
        QObject::connect(action, SIGNAL(triggered()), this, SLOT(OnHelpAbout()));
      }
      else if (action->text() == "&Welcome")
      {
        action->setVisible(false);
      }
      else if (action->text() == "&Reset Perspective")
      {
        berry::CommandContributionItemParameter::Pointer param(
              new berry::CommandContributionItemParameter(
                window.GetPointer(),
                QString(),
                berry::IWorkbenchCommandConstants::WINDOW_SAVE_PERSPECTIVE_AS,
                berry::CommandContributionItem::STYLE_PUSH));
        param->label = "Save Perspective &As...";
        m_SavePerspectiveItem = new berry::CommandContributionItem(param);
        m_SavePerspectiveItem->Fill(menu, action);

        QAction* deletePerspectiveAction = new QAction("&Delete perspective", menu);
        this->connect(deletePerspectiveAction, SIGNAL(triggered(bool)), SLOT(OnDeletePerspective()));
        menu->insertAction(action, deletePerspectiveAction);
      }
    }
  }

  m_PerspectiveListener.reset(new PerspectiveListener(window));
  window->AddPerspectiveListener(m_PerspectiveListener.data());
}


//-----------------------------------------------------------------------------
void BaseWorkbenchWindowAdvisor::OpenEditor(const QString& editorName)
{
  berry::IWorkbenchWindow::Pointer wnd = this->GetWindowConfigurer()->GetWindow();
  berry::IWorkbenchPage::Pointer page = wnd->GetActivePage();
  ctkPluginContext* context = PluginActivator::GetInstance()->GetContext();
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
void BaseWorkbenchWindowAdvisor::OpenEditorIfEnvironmentVariableIsON(const std::string& envVariable, const QString& editorName)
{
  const char* envVarValue = std::getenv(envVariable.c_str());
  if (envVarValue && std::strcmp(envVarValue, "ON") == 0)
  {
    this->OpenEditor(editorName);
  }
}

}

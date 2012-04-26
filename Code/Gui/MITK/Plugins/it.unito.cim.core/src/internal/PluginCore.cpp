/*=============================================================================

 KMaps:     An image processing toolkit for DCE-MRI analysis developed
            at the Molecular Imaging Center at University of Torino.

 See:       http://www.cim.unito.it

 Author:    Miklos Espak <espakm@gmail.com>

 Copyright (c) Miklos Espak
 All Rights Reserved.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "PluginCore.h"

#include <berryIWorkbenchWindow.h>
#include <berryISelectionService.h>

#include <berryPlatformUI.h>
#include <berryIWorkbench.h>
#include <berryIWorkbenchWindow.h>
#include <berryIWorkbenchPage.h>
#include <berryIWorkbenchPart.h>
#include <berryISelectionListener.h>

#include <mitkDataNodeObject.h>
#include <mitkIDataStorageService.h>
#include <mitkDataStorageEditorInput.h>
#include <mitkWeakPointer.h>

//#include <QmitkStdMultiWidgetEditor.h>

#include <QSet>

#include "VisibilityChangedCommand.h"

class PluginCorePrivate
{
public:
  mitk::WeakPointer<mitk::DataStorage> dataStorage;

  berry::ISelectionService* selectionService;
  berry::ISelectionListener::Pointer blueBerrySelectionListener;
  berry::IWorkbenchWindow* workbenchWindow;

  QMap<const mitk::DataNode*, unsigned long> visibilityObserverTags;

  mitk::MessageDelegate1<PluginCore, const mitk::DataNode*>* addNodeEventListener;
  mitk::MessageDelegate1<PluginCore, const mitk::DataNode*>* removeNodeEventListener;
//  QSet<mitk::DataNode*> visibleNodes;
};


PluginCore::PluginCore()
: d_ptr(new PluginCorePrivate)
{
  Q_D(PluginCore);

  mitk::IDataStorageService::Pointer service =
    berry::Platform::GetServiceRegistry().GetServiceById<mitk::IDataStorageService>(mitk::IDataStorageService::ID);

  if (service.IsNotNull())
  {
    d->dataStorage = service->GetDefaultDataStorage()->GetDataStorage();

    d->addNodeEventListener =
        new mitk::MessageDelegate1<PluginCore, const mitk::DataNode*>(this, &PluginCore::onNodeAddedInternal);
    d->dataStorage->AddNodeEvent.AddListener(*d->addNodeEventListener);

    d->removeNodeEventListener =
        new mitk::MessageDelegate1<PluginCore, const mitk::DataNode*>(this, &PluginCore::onNodeRemovedInternal);
    d->dataStorage->RemoveNodeEvent.AddListener(*d->removeNodeEventListener);

//    d->dataStorage->ChangedNodeEvent.AddListener(
//        mitk::MessageDelegate1<PluginCore, const mitk::DataNode*>(this, &PluginCore::onNodeChangedInternal));
  }
  else {
    d->dataStorage = 0;
  }

  d->workbenchWindow = 0;
  d->selectionService = 0;
}

PluginCore::~PluginCore() {
  Q_D(PluginCore);

  if (d->dataStorage.IsNotNull())
  {
    foreach (const mitk::DataNode* node, d->visibilityObserverTags.keys()) {
      if (d->dataStorage->Exists(node)) {
        mitk::BaseProperty* property = node->GetProperty("visible");
        if (property) {
          property->RemoveObserver(d->visibilityObserverTags[node]);
        }
      }
    }

    d->dataStorage->AddNodeEvent.RemoveListener(*d->addNodeEventListener);
    d->dataStorage->RemoveNodeEvent.RemoveListener(*d->removeNodeEventListener);

    delete d->addNodeEventListener;
    delete d->removeNodeEventListener;
//    d->dataStorage->ChangedNodeEvent.AddListener(
//        mitk::MessageDelegate1<PluginCore, const mitk::DataNode*>(this, &PluginCore::onNodeChangedInternal));
  }
}

mitk::DataStorage*
PluginCore::GetDataStorage()
{
  Q_D(PluginCore);
  return d->dataStorage;
}

void
PluginCore::init()
{
  Q_D(PluginCore);
  if (d->workbenchWindow) {
    return;
  }
  berry::IWorkbench* workbench = berry::PlatformUI::GetWorkbench();
  if (!workbench) {
    return;
  }
  berry::IWorkbenchWindow::Pointer workbenchWindow = workbench->GetWorkbenchWindows().at(0);
  if (workbenchWindow.IsNull()) {
    return;
  }
  d->workbenchWindow = workbenchWindow.GetPointer();
  d->selectionService = workbenchWindow->GetSelectionService();

  // REGISTER FOR WORKBENCH SELECTION EVENTS
  d->blueBerrySelectionListener = new berry::SelectionChangedAdapter<PluginCore>(this, &PluginCore::BlueBerrySelectionChanged);
  d->selectionService->AddPostSelectionListener(/*"org.mitk.views.datamanager",*/ d->blueBerrySelectionListener);
}

//QmitkStdMultiWidget*
//PluginCore::GetActiveStdMultiWidget()
//{
//  Q_D(PluginCore);
//
//  if (!d->workbenchWindow) {
//    return 0;
//  }
//
//  QmitkStdMultiWidget* activeStdMultiWidget = 0;
//  berry::IWorkbenchPage::Pointer activePage = d->workbenchWindow->GetActivePage();
//  if (activePage.IsNull()) {
//    return 0;
//  }
//  berry::IEditorPart::Pointer editor = activePage->GetActiveEditor();
//
//  if (editor.Cast<QmitkStdMultiWidgetEditor>().IsNotNull()) {
//    activeStdMultiWidget = editor.Cast<QmitkStdMultiWidgetEditor>()->GetStdMultiWidget();
//  }
//
//  return activeStdMultiWidget;
//}

void
PluginCore::onNodeAddedInternal(const mitk::DataNode* node)
{
  Q_D(PluginCore);
  if (!d->workbenchWindow) {
    init();
  }

  mitk::BaseProperty* property = node->GetProperty("visible");
  if (property) {
//    mitk::BoolProperty* visibilityProperty = dynamic_cast<mitk::BoolProperty*>(property);
//    if (visibilityProperty && visibilityProperty->GetValue()) {
//      d->visibleNodes.insert(node);
//    }
    VisibilityChangedCommand::Pointer command = VisibilityChangedCommand::New(this, node);
    d->visibilityObserverTags[node] = property->AddObserver(itk::ModifiedEvent(), command);
  }

  onNodeAdded(node);
}

void
PluginCore::onNodeRemovedInternal(const mitk::DataNode* node)
{
  Q_D(PluginCore);
  if (d->visibilityObserverTags.contains(node)) {
    mitk::BaseProperty* property = node->GetProperty("visible");
    if (property) {
//      mitk::BoolProperty* visibilityProperty = dynamic_cast<mitk::BoolProperty*>(property);
//      if (!visibilityProperty || !visibilityProperty->GetValue()) {
//        d->visibleNodes.remove(node);
//      }
      property->RemoveObserver(d->visibilityObserverTags[node]);
    }
    d->visibilityObserverTags.remove(node);
  }

  onNodeRemoved(node);
}

//void
//PluginCore::onVisibilityChangedInternal(const mitk::DataNode* node)
//{
//  Q_D(PluginCore);
//  mitk::BaseProperty* property = node->GetProperty("visible");
//  if (property) {
//    mitk::BoolProperty* visibilityProperty = dynamic_cast<mitk::BoolProperty*>(property);
//    if (visibilityProperty && visibilityProperty->GetValue()) {
//      d->visibleNodes.insert(node);
//    }
//    else {
//      d->visibleNodes.remove(node);
//    }
//  }
//
//  onVisibilityChanged(node);
//}

void
PluginCore::onNodeAdded(const mitk::DataNode* /*node*/)
{
}

void
PluginCore::onNodeRemoved(const mitk::DataNode* /*node*/)
{
}

//void
//PluginCore::onNodeChanged(const mitk::DataNode* /*node*/)
//{
//}

void
PluginCore::onVisibilityChanged(const mitk::DataNode* /*node*/)
{
}

void
PluginCore::BlueBerrySelectionChanged(berry::IWorkbenchPart::Pointer sourcepart, berry::ISelection::ConstPointer selection)
{
  if(sourcepart.IsNull() || sourcepart->GetSite()->GetId() != "org.mitk.views.datamanager")
    return;

  Q_D(PluginCore);
  if (!d->workbenchWindow) {
    init();
  }

  mitk::DataNodeSelection::ConstPointer _DataNodeSelection = selection.Cast<const mitk::DataNodeSelection>();
  this->OnSelectionChanged(this->DataNodeSelectionToVector(_DataNodeSelection));
}

std::vector<mitk::DataNode*>
PluginCore::GetCurrentSelection() const
{
  Q_D(const PluginCore);
  berry::ISelection::ConstPointer selection( d->selectionService->GetSelection());
  // buffer for the data manager selection
  mitk::DataNodeSelection::ConstPointer currentSelection = selection.Cast<const mitk::DataNodeSelection>();
  return this->DataNodeSelectionToVector(currentSelection);
}

std::vector<mitk::DataNode*>
PluginCore::GetDataManagerSelection() const
{
  Q_D(const PluginCore);
  berry::ISelection::ConstPointer selection( d->selectionService->GetSelection("org.mitk.views.datamanager"));
    // buffer for the data manager selection
  mitk::DataNodeSelection::ConstPointer currentSelection = selection.Cast<const mitk::DataNodeSelection>();
  return this->DataNodeSelectionToVector(currentSelection);
}

void
PluginCore::OnSelectionChanged(std::vector<mitk::DataNode*> /*nodes*/)
{
}

std::vector<mitk::DataNode*>
PluginCore::DataNodeSelectionToVector(mitk::DataNodeSelection::ConstPointer currentSelection) const
{
  std::vector<mitk::DataNode*> selectedNodes;
  if(currentSelection.IsNull())
    return selectedNodes;

  mitk::DataNodeObject* _DataNodeObject = 0;
  mitk::DataNode* _DataNode = 0;

  for(mitk::DataNodeSelection::iterator it = currentSelection->Begin();
    it != currentSelection->End(); ++it)
  {
    _DataNodeObject = dynamic_cast<mitk::DataNodeObject*>((*it).GetPointer());
    if(_DataNodeObject)
    {
      _DataNode = _DataNodeObject->GetDataNode();
      if(_DataNode)
        selectedNodes.push_back(_DataNode);
    }
  }

  return selectedNodes;
}

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

#include "FunctionalityBase.h"

#include <itkCommand.h>

#include "internal/VisibilityChangedCommand.h"

class FunctionalityBasePrivate
{
public:
  QMap<const mitk::DataNode*, unsigned long> visibilityObserverTags;

  mitk::MessageDelegate1<FunctionalityBase, const mitk::DataNode*>* addNodeEventListener;
  mitk::MessageDelegate1<FunctionalityBase, const mitk::DataNode*>* removeNodeEventListener;
};


FunctionalityBase::FunctionalityBase()
: QmitkAbstractView(),
  d_ptr(new FunctionalityBasePrivate)
{
  mitk::DataStorage* dataStorage = GetDataStorage();

  if (dataStorage) {

    mitk::DataStorage::SetOfObjects::ConstPointer everyNode = dataStorage->GetAll();
    mitk::DataStorage::SetOfObjects::ConstIterator it = everyNode->Begin();
    mitk::DataStorage::SetOfObjects::ConstIterator end = everyNode->End();
    while (it != end) {
      onNodeAddedInternal(it->Value());
      ++it;
    }

    Q_D(FunctionalityBase);
    d->addNodeEventListener =
        new mitk::MessageDelegate1<FunctionalityBase, const mitk::DataNode*>(this, &FunctionalityBase::onNodeAddedInternal);
    dataStorage->AddNodeEvent.AddListener(*d->addNodeEventListener);

    d->removeNodeEventListener =
        new mitk::MessageDelegate1<FunctionalityBase, const mitk::DataNode*>(this, &FunctionalityBase::onNodeRemovedInternal);
    dataStorage->RemoveNodeEvent.AddListener(*d->removeNodeEventListener);
  }
  else {
    MITK_INFO << "FunctionalityBase() data storage not ready";
  }
}

FunctionalityBase::~FunctionalityBase() {
  Q_D(FunctionalityBase);

  mitk::DataStorage* dataStorage = GetDataStorage();
  if (dataStorage)
  {
    dataStorage->AddNodeEvent.RemoveListener(*d->addNodeEventListener);
    dataStorage->RemoveNodeEvent.RemoveListener(*d->removeNodeEventListener);

    delete d->addNodeEventListener;
    delete d->removeNodeEventListener;
//    d->dataStorage->ChangedNodeEvent.AddListener(
//        mitk::MessageDelegate1<PluginCore, const mitk::DataNode*>(this, &PluginCore::onNodeChangedInternal));
  }

  foreach (const mitk::DataNode* node, d->visibilityObserverTags.keys()) {
    mitk::BaseProperty* property = node->GetProperty("visible");
    if (property) {
      property->RemoveObserver(d->visibilityObserverTags[node]);
    }
  }
}

bool
FunctionalityBase::IsExclusiveFunctionality() const
{
  return false;
}

void
FunctionalityBase::onNodeAddedInternal(const mitk::DataNode* node)
{
  Q_D(FunctionalityBase);

  mitk::BaseProperty* property = node->GetProperty("visible");
  if (property) {
    VisibilityChangedCommand::Pointer command = VisibilityChangedCommand::New(this, node);
    d->visibilityObserverTags[node] = property->AddObserver(itk::ModifiedEvent(), command);
  }
}

void
FunctionalityBase::onNodeRemovedInternal(const mitk::DataNode* node)
{
  Q_D(FunctionalityBase);
  if (d->visibilityObserverTags.contains(node)) {
    mitk::BaseProperty* property = node->GetProperty("visible");
    if (property) {
      property->RemoveObserver(d->visibilityObserverTags[node]);
    }
    d->visibilityObserverTags.remove(node);
  }
}

void
FunctionalityBase::onVisibilityChanged(const mitk::DataNode* /*node*/)
{
}

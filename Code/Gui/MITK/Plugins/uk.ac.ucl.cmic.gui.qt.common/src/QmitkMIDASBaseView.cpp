/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-11-24 15:53:45 +0000 (Thu, 24 Nov 2011) $
 Revision          : $Revision: 7857 $
 Last modified by  : $Author: mjc $

 Original author   : Miklos Espak <espakm@gmail.com>

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#include "QmitkMIDASBaseView.h"

#include <itkCommand.h>

#include "internal/VisibilityChangedCommand.h"

class QmitkMIDASBaseViewPrivate
{
public:
  QMap<const mitk::DataNode*, unsigned long> visibilityObserverTags;

  mitk::MessageDelegate1<QmitkMIDASBaseView, const mitk::DataNode*>* addNodeEventListener;
  mitk::MessageDelegate1<QmitkMIDASBaseView, const mitk::DataNode*>* removeNodeEventListener;
};


QmitkMIDASBaseView::QmitkMIDASBaseView()
: QmitkAbstractView(),
  d_ptr(new QmitkMIDASBaseViewPrivate)
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

    Q_D(QmitkMIDASBaseView);
    d->addNodeEventListener =
        new mitk::MessageDelegate1<QmitkMIDASBaseView, const mitk::DataNode*>(this, &QmitkMIDASBaseView::onNodeAddedInternal);
    dataStorage->AddNodeEvent.AddListener(*d->addNodeEventListener);

    d->removeNodeEventListener =
        new mitk::MessageDelegate1<QmitkMIDASBaseView, const mitk::DataNode*>(this, &QmitkMIDASBaseView::onNodeRemovedInternal);
    dataStorage->RemoveNodeEvent.AddListener(*d->removeNodeEventListener);
  }
  else {
    MITK_INFO << "QmitkMIDASBaseView() data storage not ready";
  }
}

QmitkMIDASBaseView::~QmitkMIDASBaseView() {
  Q_D(QmitkMIDASBaseView);

  mitk::DataStorage* dataStorage = GetDataStorage();
  if (dataStorage)
  {
    dataStorage->AddNodeEvent.RemoveListener(*d->addNodeEventListener);
    dataStorage->RemoveNodeEvent.RemoveListener(*d->removeNodeEventListener);

    delete d->addNodeEventListener;
    delete d->removeNodeEventListener;
  }

  foreach (const mitk::DataNode* node, d->visibilityObserverTags.keys()) {
    mitk::BaseProperty* property = node->GetProperty("visible");
    if (property) {
      property->RemoveObserver(d->visibilityObserverTags[node]);
    }
  }
}

bool
QmitkMIDASBaseView::IsExclusiveFunctionality() const
{
  return false;
}

void
QmitkMIDASBaseView::onNodeAddedInternal(const mitk::DataNode* node)
{
  Q_D(QmitkMIDASBaseView);

  mitk::BaseProperty* property = node->GetProperty("visible");
  if (property) {
    VisibilityChangedCommand::Pointer command = VisibilityChangedCommand::New(this, node);
    d->visibilityObserverTags[node] = property->AddObserver(itk::ModifiedEvent(), command);
  }
}

void
QmitkMIDASBaseView::onNodeRemovedInternal(const mitk::DataNode* node)
{
  Q_D(QmitkMIDASBaseView);
  if (d->visibilityObserverTags.contains(node)) {
    mitk::BaseProperty* property = node->GetProperty("visible");
    if (property) {
      property->RemoveObserver(d->visibilityObserverTags[node]);
    }
    d->visibilityObserverTags.remove(node);
  }
}

void
QmitkMIDASBaseView::onVisibilityChanged(const mitk::DataNode* /*node*/)
{
}

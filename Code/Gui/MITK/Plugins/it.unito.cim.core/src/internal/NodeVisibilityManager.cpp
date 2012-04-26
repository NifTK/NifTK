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

#include "NodeVisibilityManager.h"

#include <cmath>

#include <itkSimpleDataObjectDecorator.h>

#include <mitkRenderingManager.h>
#include <mitkNodePredicateNot.h>
#include <mitkNodePredicateAnd.h>
#include <mitkNodePredicateData.h>
#include <mitkNodePredicateProperty.h>
#include <mitkNodePredicateSource.h>
#include <mitkGenericProperty.h>

class NodeVisibilityManagerPrivate
{
public:
  mitk::DataStorage::Pointer dataStorage;
  mitk::NodePredicateNot::Pointer pred;

  std::vector<mitk::DataNode*> nodes;
};


NodeVisibilityManager::NodeVisibilityManager()
: d_ptr(new NodeVisibilityManagerPrivate)
{
  Q_D(NodeVisibilityManager);

  d->dataStorage = GetDataStorage();
}

NodeVisibilityManager::~NodeVisibilityManager()
{
}

void
NodeVisibilityManager::onNodeAdded(const mitk::DataNode* node)
{
  if (node->IsVisible(0)) {
    propagateVisibilityOn(node);
  }
}

void
NodeVisibilityManager::onNodeRemoved(const mitk::DataNode* node)
{
  if (node->IsVisible(0) && !node->GetData()) {
    propagateVisibilityOff(node);
  }
}

void
NodeVisibilityManager::onVisibilityChanged(const mitk::DataNode* node)
{
  if (node->IsVisible(0)) {
    propagateVisibilityOn(node);
  }
  else if (!node->GetData()) {
    propagateVisibilityOff(node);
  }
}

void
NodeVisibilityManager::propagateVisibilityOn(const mitk::DataNode* node) {
  Q_D(NodeVisibilityManager);

  static mitk::NodePredicateData::Pointer isGroupingNode =
      mitk::NodePredicateData::New(0);
  static mitk::NodePredicateProperty::Pointer isNotVisible =
      mitk::NodePredicateProperty::New("visible", mitk::BoolProperty::New(false));
  static mitk::NodePredicateAnd::Pointer notVisibleGroupingNode =
      mitk::NodePredicateAnd::New(isGroupingNode, isNotVisible);

  mitk::DataStorage::SetOfObjects::ConstPointer notVisibleGroupingNodeSources =
      d->dataStorage->GetSources(node, notVisibleGroupingNode, false);

  mitk::DataStorage::SetOfObjects::const_iterator it = notVisibleGroupingNodeSources->begin();
  mitk::DataStorage::SetOfObjects::const_iterator end = notVisibleGroupingNodeSources->end();
  while (it != end) {
    (*it)->SetVisibility(true);
    ++it;
  }
}

void
NodeVisibilityManager::propagateVisibilityOff(const mitk::DataNode* node) {
  Q_D(NodeVisibilityManager);

  static mitk::NodePredicateProperty::Pointer isVisible =
      mitk::NodePredicateProperty::New("visible", mitk::BoolProperty::New(true));

  mitk::DataStorage::SetOfObjects::ConstPointer visibleDerivants =
      d->dataStorage->GetDerivations(node, isVisible, false);

  mitk::DataStorage::SetOfObjects::const_iterator it = visibleDerivants->begin();
  mitk::DataStorage::SetOfObjects::const_iterator end = visibleDerivants->end();
  while (it != end) {
    (*it)->SetVisibility(false);
    ++it;
  }
}

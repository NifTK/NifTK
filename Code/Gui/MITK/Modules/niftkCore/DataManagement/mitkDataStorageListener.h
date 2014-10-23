/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkDataStorageListener_h
#define mitkDataStorageListener_h

#include "niftkCoreExports.h"
#include <itkObject.h>
#include <mitkDataStorage.h>
#include <mitkDataNode.h>
#include "mitkDataNodeFilter.h"

namespace mitk
{

/**
 * \class DataStorageListener
 * \brief Base class for objects that Listen to data storage, and want to update a node.
 *
 * Derived classes must override NodeAdded, NodeUpdated, NodeRemoved, NodeDeleted.
 * This class also provides a filter mechanism, where a chain of filters can be added,
 * and if any one of the filters blocks, then the methods NodeAdded, NodeUpdated,
 * NodeRemoved and NodeDeleted are not called.
 */
class NIFTKCORE_EXPORT DataStorageListener : public itk::LightObject
{

public:

  mitkClassMacro(DataStorageListener, itk::LightObject);
  mitkNewMacro1Param(DataStorageListener, const mitk::DataStorage::Pointer);

  /// \brief Gets the data storage.
  mitk::DataStorage::Pointer GetDataStorage() const;

  /// \brief Adds a filter.
  ///
  /// \param filter A subclass of mitk::DataNodeFilter.
  void AddFilter(mitk::DataNodeFilter::Pointer filter);

  /// \brief Clears all filters.
  void ClearFilters();

  /// \brief Tells if the listener is blocked, i.e. the processing of signals is suppressed.
  bool IsBlocked() const;

  /// \brief Blocks the processing of signals.
  /// This can be used to avoid eventual infinite recursion.
  /// Returns true if the listener was already blocked, otherwise false.
  bool SetBlocked(bool blocked);
  
  /// \brief GUI independent message callback.
  Message1<mitk::DataNode*> NodeAdded;
  Message1<mitk::DataNode*> NodeChanged;
  Message1<mitk::DataNode*> NodeRemoved;
  Message1<mitk::DataNode*> NodeDeleted;

protected:

  DataStorageListener(const mitk::DataStorage::Pointer);
  virtual ~DataStorageListener();

  DataStorageListener(const DataStorageListener&); // Purposefully not implemented.
  DataStorageListener& operator=(const DataStorageListener&); // Purposefully not implemented.

  /// \brief Called when the given node is added to the data storage.
  /// Empty implementation, subclasses can redefine it.
  virtual void OnNodeAdded(mitk::DataNode* node);

  /// \brief Called when the given node has been changed.
  /// Empty implementation, subclasses can redefine it.
  virtual void OnNodeChanged(mitk::DataNode* node);

  /// \brief Called when the given node is removed from the data storage.
  /// Empty implementation, subclasses can redefine it.
  virtual void OnNodeRemoved(mitk::DataNode* node);

  /// \brief Called when the given node is deleted.
  /// Empty implementation, subclasses can redefine it.
  virtual void OnNodeDeleted(mitk::DataNode* node);

private:

  /// \brief Called to register to the data storage.
  void AddListeners();

  /// \brief Called to un-register from the data storage.
  void RemoveListeners();

  /// \brief Checks the node against the list of filters.
  ///
  /// \param node A data node to check
  /// \return true if the data node passess all filters and false otherwise.
  bool Pass(const mitk::DataNode* node) const;

  /// \brief Called when a DataStorage AddNodeEvent was emmitted and calls NodeAdded afterwards,
  /// and subclasses should override the NodeAdded event.
  virtual void NodeAddedProxy(const mitk::DataNode* node);

  /// \brief Called when a DataStorage ChangedNodeEvent was emmitted and calls NodeUpdated afterwards,
  /// and subclasses should override the NodeUpdated event.
  virtual void NodeChangedProxy(const mitk::DataNode* node);

  /// \brief Called when a DataStorage RemoveNodeEvent was emmitted and calls NodeRemoved afterwards,
  /// and subclasses should override the NodeRemoved event.
  virtual void NodeRemovedProxy(const mitk::DataNode* node);

  /// \brief Called when a DataStorage DeleteNodeEvent was emmitted and calls NodeDeleted afterwards,
  /// and subclasses should override the NodeDeleted event.
  virtual void NodeDeletedProxy(const mitk::DataNode* node);

  /// \brief  This object MUST be connected to a datastorage for it to work.
  mitk::DataStorage::Pointer m_DataStorage;

  /// \brief Simply keeps track of whether we are currently processing an update to avoid repeated/recursive calls.
  bool m_InDataStorageChanged;

  /// \brief We maintain a list of filters that can stop the derived methods being called.
  std::vector<mitk::DataNodeFilter*> m_Filters;

  /// \brief Maintain a boolean to blocked calling derived class methods NodeAdded, NodeChanged, NodeRemoved, NodeDeleted etc.
  bool m_Blocked;
};

} // end namespace

#endif

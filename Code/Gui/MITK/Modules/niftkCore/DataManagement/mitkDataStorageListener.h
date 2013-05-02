/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitk_DataStorageListener_h
#define mitk_DataStorageListener_h

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
class NIFTKCORE_EXPORT DataStorageListener : public itk::Object
{

public:

  mitkClassMacro(DataStorageListener, itk::Object);
  itkNewMacro(DataStorageListener);
  mitkNewMacro1Param(DataStorageListener, const mitk::DataStorage::Pointer);

  /// \brief Get the data storage.
  itkGetMacro(DataStorage, mitk::DataStorage::Pointer);

  /// \brief Set the data storage.
  void SetDataStorage(const mitk::DataStorage::Pointer dataStorage);

  /// \brief Adds a filter.
  ///
  /// \param filter A subclass of mitk::DataNodeFilter.
  void AddFilter(mitk::DataNodeFilter::Pointer filter);

  /// \brief Clears all filters.
  void ClearFilters();

  /// \brief Set/Get the block variable to determine if we are blocking the NodeAdded, NodeChanged etc calls. Default is false.
  itkGetMacro(Block, bool);
  itkSetMacro(Block, bool);

protected:

  DataStorageListener();
  DataStorageListener(const mitk::DataStorage::Pointer);
  virtual ~DataStorageListener();

  DataStorageListener(const DataStorageListener&); // Purposefully not implemented.
  DataStorageListener& operator=(const DataStorageListener&); // Purposefully not implemented.

  /// \brief Called to register to the data storage.
  virtual void Activate(const mitk::DataStorage::Pointer storage);

  /// \brief Called to un-register from the data storage.
  virtual void Deactivate();

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

  /// \brief In this class, we do nothing, as subclasses should re-define this.
  virtual void NodeAdded(mitk::DataNode* node) {};

  /// \brief In this class, we do nothing, as subclasses should re-define this.
  virtual void NodeChanged(mitk::DataNode* node) {};

  /// \brief In this class, we do nothing, as subclasses should re-define this.
  virtual void NodeRemoved(mitk::DataNode* node) {};

  /// \brief In this class, we do nothing, as subclasses should re-define this.
  virtual void NodeDeleted(mitk::DataNode* node) {};

private:

  /// \brief Checks the node against the list of filters.
  ///
  /// \param node A data node to check
  /// \return true if the data node passess all filters and false otherwise.
  bool Pass(const mitk::DataNode* node);

  /// \brief  This object MUST be connected to a datastorage for it to work.
  mitk::DataStorage::Pointer m_DataStorage;

  /// \brief Simply keeps track of whether we are currently processing an update to avoid repeated/recursive calls.
  bool m_InDataStorageChanged;

  /// \brief We maintain a list of filters that can stop the derived methods being called.
  std::vector<mitk::DataNodeFilter*> m_Filters;

  /// \brief Maintain a boolean to block calling derived class methods NodeAdded, NodeChanged, NodeRemoved, NodeDeleted etc.
  bool m_Block;
};

} // end namespace

#endif

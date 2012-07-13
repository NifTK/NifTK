/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-12-16 09:12:58 +0000 (Fri, 16 Dec 2011) $
 Revision          : $Revision: 8039 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKMIDASDATASTORAGELISTENER_H_
#define MITKMIDASDATASTORAGELISTENER_H_

#include "niftkMitkExtExports.h"

#include <mitkDataStorage.h>
#include <mitkDataNode.h>

namespace mitk
{

/**
 * \class DataStorageListener
 * \brief Base class for objects that Listen to data storage, and want to update a node.
 *
 * Derived classes must override NodeAdded, NodeUpdated, NodeRemoved, NodeDeleted.
 */
class NIFTKMITKEXT_EXPORT DataStorageListener
{

public:

  /// \brief This class must (checked with assert) have a non-NULL mitk::DataStorage
  /// so it is injected in the constructor, and we register to AddNodeEvent, RemoveNodeEvent.
  DataStorageListener(mitk::DataStorage::Pointer dataStorage);

  /// \brief Destructor, which unregisters all the listeners.
  virtual ~DataStorageListener();

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

protected:

  /// \brief In this class, we do nothing, as subclasses should re-define this.
  virtual void NodeAdded(mitk::DataNode* node) {};

  /// \brief In this class, we do nothing, as subclasses should re-define this.
  virtual void NodeChanged(mitk::DataNode* node) {};

  /// \brief In this class, we do nothing, as subclasses should re-define this.
  virtual void NodeRemoved(mitk::DataNode* node) {};

  /// \brief In this class, we do nothing, as subclasses should re-define this.
  virtual void NodeDeleted(mitk::DataNode* node) {};

  /// \brief  This object MUST be connected to a datastorage, hence it is passed in via the constructor.
  mitk::DataStorage::Pointer m_DataStorage;

private:

  /// \brief Simply keeps track of whether we are currently processing an update to avoid repeated/recursive calls.
  bool m_InDataStorageChanged;
};

} // end namespace

#endif

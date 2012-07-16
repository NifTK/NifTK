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

#include <itkObject.h>
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
class NIFTKMITKEXT_EXPORT DataStorageListener : public itk::Object
{

public:

  mitkClassMacro(DataStorageListener, itk::Object);
  itkNewMacro(DataStorageListener);
  mitkNewMacro1Param(DataStorageListener, const mitk::DataStorage::Pointer);

  /// \brief Get the data storage.
  itkGetMacro(DataStorage, mitk::DataStorage::Pointer);

  /// \brief Set the data storage.
  void SetDataStorage(const mitk::DataStorage::Pointer dataStorage);

protected:

  DataStorageListener();
  DataStorageListener(const mitk::DataStorage::Pointer);
  virtual ~DataStorageListener();

  DataStorageListener(const DataStorageListener&); // Purposefully not implemented.
  DataStorageListener& operator=(const DataStorageListener&); // Purposefully not implemented.

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

  /// \brief Called to register to the data storage.
  void Activate(const mitk::DataStorage::Pointer storage);

  /// \brief Called to un-register from the data storage.
  void Deactivate();

  /// \brief  This object MUST be connected to a datastorage for it to work.
  mitk::DataStorage::Pointer m_DataStorage;

  /// \brief Simply keeps track of whether we are currently processing an update to avoid repeated/recursive calls.
  bool m_InDataStorageChanged;
};

} // end namespace

#endif

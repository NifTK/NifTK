/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKMIDASDATASTORAGEPROPERTYLISTENER_H_
#define MITKMIDASDATASTORAGEPROPERTYLISTENER_H_

#include "niftkCoreExports.h"
#include "mitkDataStorageListener.h"
#include <itkObject.h>
#include <mitkDataStorage.h>
#include <mitkDataNode.h>
#include <mitkMessage.h>

namespace mitk
{

/**
 * \class DataStoragePropertyListener
 * \brief Base class for objects that Listen to data storage for a specific property such as "visibility".
 *
 * This class is derived from itk::Object so we can use things like the ITK setter/getter macros, listening to
 * Modified events via the Observer pattern etc.
 *
 * Derived classes must implement OnPropertyChanged.
 */
class NIFTKCORE_EXPORT DataStoragePropertyListener : public mitk::DataStorageListener
{

public:

  mitkClassMacro(DataStoragePropertyListener, mitk::DataStorageListener);
  itkNewMacro(DataStoragePropertyListener);
  mitkNewMacro1Param(DataStoragePropertyListener, const mitk::DataStorage::Pointer);

  /// \brief Get the property name.
  itkGetMacro(PropertyName, std::string);

  /// \brief Set the property name, which triggers an update UpdateObserverToPropertyMap.
  void SetPropertyName(const std::string& name);

  /// \brief Sets the list of renderers to check.
  void SetRenderers(std::vector<mitk::BaseRenderer*>& list);

  /// \brief Clears all filters.
  void ClearRenderers();

  /// \brief GUI independent message callback.
  Message<> PropertyChanged;

protected:

  DataStoragePropertyListener();
  DataStoragePropertyListener(const mitk::DataStorage::Pointer);
  virtual ~DataStoragePropertyListener();

  DataStoragePropertyListener(const DataStoragePropertyListener&); // Purposefully not implemented.
  DataStoragePropertyListener& operator=(const DataStoragePropertyListener&); // Purposefully not implemented.

  /// \brief Called to register to the data storage.
  virtual void Activate(const mitk::DataStorage::Pointer storage);

  /// \brief Called to un-register from the data storage.
  virtual void Deactivate();

  /// \brief In this class, we do nothing, as subclasses should re-define this.
  virtual void OnPropertyChanged(const itk::EventObject&);

  /// \brief Will refresh the observers of the named property, and sub-classes should call this at the appropriate time.
  virtual void UpdateObserverToPropertyMap();

  /// \brief Will remove all observers from the m_ObserverToPropertyMap, and sub-classes should call this at the appropriate time.
  virtual void RemoveAllFromObserverToPropertyMap();

  /// \brief Triggers UpdateObserverToPropertyMap.
  ///
  /// \see DataStoragePropertyListener::NodeAdded
  virtual void NodeAdded(mitk::DataNode* node);

  /// \brief Triggers UpdateObserverToPropertyMap.
  ///
  /// \see DataStoragePropertyListener::NodeAdded
  virtual void NodeRemoved(mitk::DataNode* node);

  /// \brief Triggers UpdateObserverToPropertyMap.
  //
  /// \see DataStoragePropertyListener::NodeAdded
  virtual void NodeDeleted(mitk::DataNode* node);

private:

  /// \brief Internal method to fire the property changed signal.
  void OnPropertyChanged();

  /// \brief We observe all the properties with a given name for each registered node.
  typedef std::pair < mitk::BaseProperty*, unsigned long > PropertyToObserver;
  typedef std::pair < mitk::BaseProperty*, mitk::DataNode* > PropertyToNode;
  typedef std::vector< PropertyToObserver > VectorPropertyToObserver;
  typedef std::vector< PropertyToNode > VectorPropertyToNode;
  VectorPropertyToObserver m_WatchedObservers;
  VectorPropertyToNode m_WatchedNodes;

  /// \brief The name of the property we are tracking.
  std::string m_PropertyName;

  /// \brief We store an optional list of renderers for watching renderer specific changes.
  std::vector<mitk::BaseRenderer*> m_Renderers;

  /// \brief Set this so that each time this class recalculates the list of properties to
  /// track, we also set each property to modified. Default false.
  bool m_AutoFire;

};

} // end namespace

#endif

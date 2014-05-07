/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkDataStoragePropertyListener_h
#define mitkDataStoragePropertyListener_h

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
 *
 * The event listening can be restricted to a set of renderers. It is the resposibility of the user of this
 * class to remove the renderer from the DataStoragePropertyListener objects when the renderer is deleted.
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
  void SetRenderers(const std::vector<mitk::BaseRenderer*>& renderers);

  /// \brief GUI independent message callback.
  Message2<mitk::DataNode*, mitk::BaseRenderer*> PropertyChanged;

  /// \brief Called when the global or a renderer specific property of the node has changed or removed.
  void OnPropertyChanged(mitk::DataNode* node, mitk::BaseRenderer* renderer);

  /// \brief Sends a signal with current the property value of the given node to the registered listeners.
  void Notify(mitk::DataNode* node);

  /// \brief Sends a signal with current the property value of all nodes to the registered listeners.
  void NotifyAll();

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

  /// \brief Will refresh the observers of the named property, and sub-classes should call this at the appropriate time.
  virtual void AddAllObservers();

  /// \brief Will remove all observers from the m_ObserverToPropertyMap, and sub-classes should call this at the appropriate time.
  virtual void RemoveAllObservers();

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

  void AddObservers(mitk::DataNode* node);
  void RemoveObservers(mitk::DataNode* node);

  typedef std::map<mitk::DataNode*, std::vector<unsigned long> > NodePropertyObserverTags;

  /// \brief We observe all the properties with a given name for each registered node.
  /// The first element of the vector is the "global" property, the rest are the renderer
  /// specific properties in the same order as in m_Renderers.
  /// The observers are notified when a property of a node is changed or removed.
  NodePropertyObserverTags m_PropertyObserverTagsPerNode;

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

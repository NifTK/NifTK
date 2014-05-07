/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkDataStorageVisibilityTracker_h
#define mitkDataStorageVisibilityTracker_h

#include "niftkCoreExports.h"
#include "mitkDataStoragePropertyListener.h"
#include <itkObject.h>
#include <mitkDataStorage.h>
#include <mitkDataNode.h>

namespace mitk
{

class BaseRenderer;

/**
 * \class DataStorageVisibilityTracker
 * \brief Observes the visibility changes of data nodes in a 'tracked' renderer and updates their visibility in the 'managed' renderers accordingly.
 *
 * If the tracked renderer is set to 0, only the global visibility is observed.
 *
 * This finds use in the Thumbnail window plugin, which tracks visibility properties, and applies
 * them to a single render window, and also the MIDAS Segmentation Viewer widget which tracks
 * visibility properties, and applies them to another viewer.
 */
class NIFTKCORE_EXPORT DataStorageVisibilityTracker : public itk::Object
{

public:

  mitkClassMacro(DataStorageVisibilityTracker, itk::Object);
  itkNewMacro(DataStorageVisibilityTracker);
  mitkNewMacro1Param(DataStorageVisibilityTracker, const mitk::DataStorage::Pointer);

  /// \brief Set the data storage, passing it onto the contained DataStoragePropertyListener.
  ///
  /// \see DataStorageListener::SetDataStorage
  void SetDataStorage(const mitk::DataStorage::Pointer dataStorage);

  /// \brief Sets the renderer we are tracking.
  void SetTrackedRenderer(mitk::BaseRenderer* trackedRenderer);

  /// \brief Sets the list of renderers to propagate visibility properties onto.
  void SetManagedRenderers(const std::vector<mitk::BaseRenderer*>& managedRenderers);

  /// \brief We provide facility to ignore nodes, and not adjust their visibility, which is useful for cross hairs.
  void SetNodesToIgnore(const std::vector<mitk::DataNode*>& nodesToIgnore);

  bool IsIgnored(mitk::DataNode* node);

  /// \brief Called when the property value has changed globally or for the given renderer.
  /// If the global property has changed, renderer is NULL.
  void OnPropertyChanged(mitk::DataNode* node, mitk::BaseRenderer* renderer);

  /// \brief Sends a signal with current the property value of all nodes  to the registered listeners.
  void NotifyAll();

protected:

  DataStorageVisibilityTracker();
  DataStorageVisibilityTracker(const mitk::DataStorage::Pointer);
  virtual ~DataStorageVisibilityTracker();

  DataStorageVisibilityTracker(const DataStorageVisibilityTracker&); // Purposefully not implemented.
  DataStorageVisibilityTracker& operator=(const DataStorageVisibilityTracker&); // Purposefully not implemented.

private:

  void Init(const mitk::DataStorage::Pointer dataStorage);

  mitk::DataStorage::Pointer m_DataStorage;
  mitk::BaseRenderer* m_TrackedRenderer;
  std::vector<mitk::BaseRenderer*> m_ManagedRenderers;
  std::vector<mitk::DataNode*> m_NodesToIgnore;

  mitk::DataStoragePropertyListener::Pointer m_Listener;
};

} // end namespace

#endif

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKMIDASDATASTORAGEVISIBILITYTRACKER_H_
#define MITKMIDASDATASTORAGEVISIBILITYTRACKER_H_

#include "niftkMitkExtExports.h"
#include "mitkDataStoragePropertyListener.h"

#include <itkObject.h>
#include <mitkDataStorage.h>
#include <mitkDataNode.h>

namespace mitk
{


class BaseRenderer;

/**
 * \class DataStorageVisibilityTracker
 * \brief Class to listen to changes in visibility properties, and to update a list of BaseRenders.
 *
 * This finds use in the MIDAS Thumbnail window, which tracks visibility properties, and applies
 * them to a single render window, and also the MIDAS Segmentation Viewer widget which tracks
 * visibility properties, and applies them to an orthoviewer.
 */
class NIFTKMITKEXT_EXPORT DataStorageVisibilityTracker : public itk::Object
{

public:

  mitkClassMacro(DataStorageVisibilityTracker, itk::Object);
  itkNewMacro(DataStorageVisibilityTracker);
  mitkNewMacro1Param(DataStorageVisibilityTracker, const mitk::DataStorage::Pointer);

  /// \brief The main Update method.
  void OnPropertyChanged();

  /// \brief Sets the list of renderers to propagate visibility properties onto.
  void SetRenderersToUpdate(std::vector<mitk::BaseRenderer*>& list);

  /// \brief Sets the renderers we are tracking.
  void SetRenderersToTrack(std::vector<mitk::BaseRenderer*>& list);

  /// \brief Set the data storage, passing it onto the contained DataStoragePropertyListener.
  ///
  /// \see DataStorageListener::SetDataStorage
  void SetDataStorage(const mitk::DataStorage::Pointer dataStorage);

  /// \brief We provide facility to ignore nodes, and not adjust their visibility, which is useful for cross hairs.
  void SetNodesToIgnore(std::vector<mitk::DataNode*>& nodes);

protected:

  DataStorageVisibilityTracker();
  DataStorageVisibilityTracker(const mitk::DataStorage::Pointer);
  virtual ~DataStorageVisibilityTracker();

  DataStorageVisibilityTracker(const DataStorageVisibilityTracker&); // Purposefully not implemented.
  DataStorageVisibilityTracker& operator=(const DataStorageVisibilityTracker&); // Purposefully not implemented.

  bool IsExcluded(mitk::DataNode* node);

private:

  void Init(const mitk::DataStorage::Pointer dataStorage);
  mitk::DataStoragePropertyListener::Pointer m_Listener;
  std::vector<mitk::BaseRenderer*> m_RenderersToTrack;
  std::vector<mitk::BaseRenderer*> m_RenderersToUpdate;
  std::vector<mitk::DataNode*> m_ExcludedNodeList;
  mitk::DataStorage::Pointer m_DataStorage;
};

} // end namespace

#endif

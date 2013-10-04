/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkDataNodeAddedVisibilitySetter_h
#define mitkDataNodeAddedVisibilitySetter_h

#include "niftkCoreExports.h"
#include <vector>
#include <mitkDataStorage.h>
#include <mitkDataStorageListener.h>

namespace mitk
{

class DataNode;
class BaseRenderer;

/**
 * \class DataNodeAddedVisibilitySetter
 * \brief When a node is added to data storage, will set initial visibility properties.
 *
 * The usage is as follows:
 * <pre>
 * 1. Call SetVisibility, specifying the default visibility.
 * 2. Call SetRenderers, specifying a list of renderers to update.
 *    If this is not set, such that m_Renderers.size() == 0, the global visibility property is updated.
 * 3. Connect to data storage.
 * </pre>
 * Thus, when a node is added to data storage, this class will listen, and automatically
 * set the default visibility property, either globally (if the renderer list is null), or
 * just for a specific list of renderers.
 *
 * \see DataStorageListener::AddFilter
 */
class NIFTKCORE_EXPORT DataNodeAddedVisibilitySetter : public DataStorageListener
{
public:

  mitkClassMacro(DataNodeAddedVisibilitySetter, DataStorageListener);
  itkNewMacro(DataNodeAddedVisibilitySetter);
  mitkNewMacro1Param(DataNodeAddedVisibilitySetter, const mitk::DataStorage::Pointer);

  /// \brief Sets the list of renderers to update.
  void SetRenderers(std::vector<mitk::BaseRenderer*>& list);

  /// \brief Clears all filters.
  void ClearRenderers();

  /// \brief Set/Get the Visibility, which defaults to false.
  itkSetMacro(Visibility, bool);
  itkGetMacro(Visibility, bool);

protected:

  DataNodeAddedVisibilitySetter();
  DataNodeAddedVisibilitySetter(const mitk::DataStorage::Pointer);
  virtual ~DataNodeAddedVisibilitySetter();

  DataNodeAddedVisibilitySetter(const DataNodeAddedVisibilitySetter&); // Purposefully not implemented.
  DataNodeAddedVisibilitySetter& operator=(const DataNodeAddedVisibilitySetter&); // Purposefully not implemented.

  /// \see DataStorageListener::NodeAdded
  virtual void NodeAdded(mitk::DataNode* node);

private:

  bool m_Visibility;
  std::vector<mitk::BaseRenderer*> m_Renderers;
};

} // end namespace

#endif

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitk_MIDASNodeAddedVisibilitySetter_h
#define mitk_MIDASNodeAddedVisibilitySetter_h

#include "niftkMIDASExports.h"
#include <vector>
#include <mitkDataStorage.h>
#include "mitkDataStorageListener.h"
#include "mitkMIDASDataNodeNameStringFilter.h"

namespace mitk
{

class DataNode;
class BaseRenderer;

/**
 * \class MIDASNodeAddedVisibilitySetter
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
class NIFTKMIDAS_EXPORT MIDASNodeAddedVisibilitySetter : public DataStorageListener
{
public:

  mitkClassMacro(MIDASNodeAddedVisibilitySetter, DataStorageListener);
  itkNewMacro(MIDASNodeAddedVisibilitySetter);
  mitkNewMacro1Param(MIDASNodeAddedVisibilitySetter, const mitk::DataStorage::Pointer);

  /// \brief Sets the list of renderers to update.
  void SetRenderers(std::vector<mitk::BaseRenderer*>& list);

  /// \brief Clears all filters.
  void ClearRenderers();

  /// \brief Set/Get the Visibility, which defaults to false.
  itkSetMacro(Visibility, bool);
  itkGetMacro(Visibility, bool);

protected:

  MIDASNodeAddedVisibilitySetter();
  MIDASNodeAddedVisibilitySetter(const mitk::DataStorage::Pointer);
  virtual ~MIDASNodeAddedVisibilitySetter();

  MIDASNodeAddedVisibilitySetter(const MIDASNodeAddedVisibilitySetter&); // Purposefully not implemented.
  MIDASNodeAddedVisibilitySetter& operator=(const MIDASNodeAddedVisibilitySetter&); // Purposefully not implemented.

  /// \see DataStorageListener::NodeAdded
  virtual void NodeAdded(mitk::DataNode* node);

private:

  bool m_Visibility;
  std::vector<mitk::BaseRenderer*> m_Renderers;
  mitk::MIDASDataNodeNameStringFilter::Pointer m_Filter;
};

} // end namespace

#endif

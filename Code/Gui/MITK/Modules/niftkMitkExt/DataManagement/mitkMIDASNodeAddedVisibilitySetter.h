/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKMIDASNODEADDEDVISIBILITYSETTER_H
#define MITKMIDASNODEADDEDVISIBILITYSETTER_H

#include "niftkMitkExtExports.h"
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
class NIFTKMITKEXT_EXPORT MIDASNodeAddedVisibilitySetter : public DataStorageListener
{
public:

  mitkClassMacro(MIDASNodeAddedVisibilitySetter, DataStorageListener);
  itkNewMacro(MIDASNodeAddedVisibilitySetter);
  mitkNewMacro1Param(MIDASNodeAddedVisibilitySetter, const mitk::DataStorage::Pointer);

  /**
   * \brief Sets the list of renderers to update.
   */
  void SetRenderers(std::vector<mitk::BaseRenderer*>& list);

  /**
   * \brief Clears all filters.
   */
  void ClearRenderers();

  /**
   * \brief Set/Get the Visibility, which defaults to false.
   */
  itkSetMacro(Visibility, bool);
  itkGetMacro(Visibility, bool);

protected:

  MIDASNodeAddedVisibilitySetter();
  MIDASNodeAddedVisibilitySetter(const mitk::DataStorage::Pointer);
  virtual ~MIDASNodeAddedVisibilitySetter();

  MIDASNodeAddedVisibilitySetter(const MIDASNodeAddedVisibilitySetter&); // Purposefully not implemented.
  MIDASNodeAddedVisibilitySetter& operator=(const MIDASNodeAddedVisibilitySetter&); // Purposefully not implemented.

  /**
   * \see DataStorageListener::NodeAdded
   */
  virtual void NodeAdded(mitk::DataNode* node);

private:

  bool m_Visibility;
  std::vector<mitk::BaseRenderer*> m_Renderers;
  mitk::MIDASDataNodeNameStringFilter::Pointer m_Filter;
};

} // end namespace

#endif

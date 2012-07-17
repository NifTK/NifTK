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

namespace mitk
{

class DataNode;
class BaseRenderer;

/**
 * \class MIDASNodeAddedVisibilitySetter
 * \brief When a node is added to data storage, will set initial visibility properties.
 */
class NIFTKMITKEXT_EXPORT MIDASNodeAddedVisibilitySetter : public DataStorageListener
{
public:

  mitkClassMacro(MIDASNodeAddedVisibilitySetter, DataStorageListener);
  itkNewMacro(MIDASNodeAddedVisibilitySetter);
  mitkNewMacro1Param(MIDASNodeAddedVisibilitySetter, const mitk::DataStorage::Pointer);

  /**
   * \brief Sets the list of renderers to check.
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

  std::vector<mitk::BaseRenderer*> m_Renderers;
  bool m_Visibility;

};

} // end namespace

#endif

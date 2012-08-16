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

#ifndef MITKDATANODEBOOLPROPERTYFILTER_H
#define MITKDATANODEBOOLPROPERTYFILTER_H

#include "niftkMitkExtExports.h"

#include "mitkDataNodeFilter.h"
#include <mitkDataStorage.h>

namespace mitk
{

class BaseRenderer;

/**
 * \class DataNodeBoolPropertyFilter
 *
 * \brief A filter that contains a list of renderers, and returns true if the node has
 * a specific boolean property set to true for those filters.
 */
class NIFTKMITKEXT_EXPORT DataNodeBoolPropertyFilter : public mitk::DataNodeFilter
{

public:

  mitkClassMacro(DataNodeBoolPropertyFilter, mitk::DataNodeFilter);
  itkNewMacro(DataNodeBoolPropertyFilter);

  /// \brief Sets the property name used for filtering.
  itkSetMacro(PropertyName, std::string);

  /// \brief Gets the property name used for filtering.
  itkGetMacro(PropertyName, std::string);

  /**
   * \brief Method to decide if the node should be passed.
   * \param node a candidate node
   * \return bool true if the node should pass and false otherwise.
   */
  virtual bool Pass(const mitk::DataNode* node);

  /**
   * \brief Sets the list of renderers to check.
   */
  void SetRenderers(std::vector<mitk::BaseRenderer*>& list);

  /**
   * \brief Sets the DataStorage against which to check.
   */
  void SetDataStorage(mitk::DataStorage::Pointer storage);

protected:

  DataNodeBoolPropertyFilter();
  virtual ~DataNodeBoolPropertyFilter();

  DataNodeBoolPropertyFilter(const DataNodeBoolPropertyFilter&); // Purposefully not implemented.
  DataNodeBoolPropertyFilter& operator=(const DataNodeBoolPropertyFilter&); // Purposefully not implemented.

private:

  std::vector<mitk::BaseRenderer*> m_Renderers;
  mitk::DataStorage::Pointer m_DataStorage;
  std::string m_PropertyName;

}; // end class

} // end namespace
#endif



/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2012-07-17 12:27:28 +0100 (Tue, 17 Jul 2012) $
 Revision          : $Revision: 9362 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKDATANODESTRINGPROPERTYFILTER_H
#define MITKDATANODESTRINGPROPERTYFILTER_H

#include "niftkMitkExtExports.h"

#include "mitkDataNodeFilter.h"

namespace mitk
{

/**
 * \class DataNodeStringPropertyFilter
 *
 * \brief A filter that takes a named property, for example "name", and a list of
 * strings to check against, and compares the node to see if its property (eg "name")
 * matches any of the supplied strings, and if it does match, will return Pass=false,
 * otherwise Pass=true.
 */
class NIFTKMITKEXT_EXPORT DataNodeStringPropertyFilter : public mitk::DataNodeFilter
{

public:

  mitkClassMacro(DataNodeStringPropertyFilter, mitk::DataNodeFilter);
  itkNewMacro(DataNodeStringPropertyFilter);

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
   * \brief Clears the list of strings to match against.
   */
  virtual void ClearList();

  /**
   * \brief Add the supplied string to the list of strings to check against.
   * \param proprtyValue a string
   */
  virtual void AddToList(const std::string& propertyValue);

  /**
   * \brief Adds a list of strings to the list of strings to check againts.
   * \param listOfStrings a list of strings
   */
  virtual void AddToList(const std::vector< std::string >& listOfStrings);

protected:

  DataNodeStringPropertyFilter();
  virtual ~DataNodeStringPropertyFilter();

  DataNodeStringPropertyFilter(const DataNodeStringPropertyFilter&); // Purposefully not implemented.
  DataNodeStringPropertyFilter& operator=(const DataNodeStringPropertyFilter&); // Purposefully not implemented.

private:

  std::string m_PropertyName;
  std::vector< std::string > m_Strings;

}; // end class

} // end namespace
#endif



/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitk_DataNodeStringPropertyFilter_h
#define mitk_DataNodeStringPropertyFilter_h

#include "niftkCoreExports.h"
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
class NIFTKCORE_EXPORT DataNodeStringPropertyFilter : public mitk::DataNodeFilter
{

public:

  mitkClassMacro(DataNodeStringPropertyFilter, mitk::DataNodeFilter);
  itkNewMacro(DataNodeStringPropertyFilter);

  /// \brief Sets the property name used for filtering.
  itkSetMacro(PropertyName, std::string);

  /// \brief Gets the property name used for filtering.
  itkGetMacro(PropertyName, std::string);

  /// \brief Method to decide if the node should be passed.
  ///
  /// \param node a candidate node
  /// \return bool true if the node should pass and false otherwise.
  virtual bool Pass(const mitk::DataNode* node);

  /// \brief Clears the list of strings to match against.
  virtual void ClearList();

  /// \brief Add the supplied string to the list of strings to check against.
  ///
  /// \param proprtyValue a string
  virtual void AddToList(const std::string& propertyValue);

  /// \brief Adds a list of strings to the list of strings to check againts.
  ///
  /// \param listOfStrings a list of strings
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



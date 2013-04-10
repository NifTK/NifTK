/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef MITKDATANODEFILTER_H
#define MITKDATANODEFILTER_H

#include "niftkCoreExports.h"
#include <mitkDataNode.h>

namespace mitk
{

/**
 * \class DataNodeFilter
 *
 * \brief A class that tests a list of nodes, and returns true for "pass" and false otherwise.
 */
class NIFTKCORE_EXPORT DataNodeFilter : public itk::Object
{

public:

  mitkClassMacro(DataNodeFilter, itk::Object);

  /// \brief Method to decide if the node should be passed.
  /// \param node a candidate node
  /// \return bool true if the node should pass and false otherwise.
  virtual bool Pass(const mitk::DataNode* node) = 0;

protected:

  DataNodeFilter() {};
  virtual ~DataNodeFilter() {};

  DataNodeFilter(const DataNodeFilter&); // Purposefully not implemented.
  DataNodeFilter& operator=(const DataNodeFilter&); // Purposefully not implemented.

}; // end class

} // end namespace
#endif



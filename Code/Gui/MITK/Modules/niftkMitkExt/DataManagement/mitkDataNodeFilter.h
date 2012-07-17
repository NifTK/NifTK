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

#ifndef MITKDATANODEFILTER_H
#define MITKDATANODEFILTER_H

#include "niftkMitkExtExports.h"

#include <mitkDataNode.h>

namespace mitk
{

/**
 * \class DataNodeFilter
 *
 * \brief A class that tests a list of nodes, and returns true for "pass" and false otherwise.
 */
class NIFTKMITKEXT_EXPORT DataNodeFilter : public itk::Object
{

public:

  mitkClassMacro(DataNodeFilter, itk::Object);

  /**
   * \brief Method to decide if the node should be passed.
   * \param node a candidate node
   * \return bool true if the node should pass and false otherwise.
   */
  virtual bool Pass(const mitk::DataNode* node) = 0;

protected:

  DataNodeFilter() {};
  virtual ~DataNodeFilter() {};

  DataNodeFilter(const DataNodeFilter&); // Purposefully not implemented.
  DataNodeFilter& operator=(const DataNodeFilter&); // Purposefully not implemented.

}; // end class

} // end namespace
#endif



/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date: 2011-08-04 08:05:37 +0100 (Thu, 04 Aug 2011) $
 Revision          : $Revision: 6968 $
 Last modified by  : $Author: mjc $

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/

#ifndef MITKDATASTORAGEUTILS_H
#define MITKDATASTORAGEUTILS_H

#include "niftkMitkExtExports.h"
#include "mitkDataNode.h"
#include "mitkDataStorage.h"

namespace mitk
{
  NIFTKMITKEXT_EXPORT bool IsNodeAGreyScaleImage(const mitk::DataNode::Pointer node);

  NIFTKMITKEXT_EXPORT bool IsNodeABinaryImage(const mitk::DataNode::Pointer node);

  NIFTKMITKEXT_EXPORT mitk::DataNode::Pointer FindFirstParentImage(const mitk::DataStorage* storage, const mitk::DataNode::Pointer node, bool lookForBinary );

  NIFTKMITKEXT_EXPORT mitk::DataNode::Pointer FindNthGreyScaleImage(const std::vector<mitk::DataNode*> &nodes, int n );

  NIFTKMITKEXT_EXPORT mitk::DataNode::Pointer FindNthBinaryImage(const std::vector<mitk::DataNode*> &nodes, int n );

  NIFTKMITKEXT_EXPORT mitk::DataNode::Pointer FindFirstGreyScaleImage(const std::vector<mitk::DataNode*> &nodes );

  NIFTKMITKEXT_EXPORT mitk::DataNode::Pointer FindFirstBinaryImage(const std::vector<mitk::DataNode*> &nodes );

  NIFTKMITKEXT_EXPORT mitk::DataNode::Pointer FindNthImage(const std::vector<mitk::DataNode*> &nodes, int n, bool lookForBinary );

  NIFTKMITKEXT_EXPORT mitk::DataStorage::SetOfObjects::Pointer FindDerivedImages(const mitk::DataStorage* storage, const mitk::DataNode::Pointer node, bool lookForBinary );

}

#endif // MITKDATASTORAGEUTILS_H

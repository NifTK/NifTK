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
#include "mitkTimeSlicedGeometry.h"
#include "mitkMIDASEnums.h"
#include "mitkMIDASImageUtils.h"

/**
 * \file mitkDataStorageUtils.h
 * \brief File containing basic DataStorage utilities such as searches, useful for a wide variety of purposes.
 */
namespace mitk
{
  NIFTKMITKEXT_EXPORT bool IsNodeAGreyScaleImage(const mitk::DataNode::Pointer node);

  NIFTKMITKEXT_EXPORT bool IsNodeABinaryImage(const mitk::DataNode::Pointer node);

  NIFTKMITKEXT_EXPORT mitk::DataNode::Pointer FindFirstParent(const mitk::DataStorage* storage, const mitk::DataNode::Pointer node);

  NIFTKMITKEXT_EXPORT mitk::DataNode::Pointer FindParentGreyScaleImage(const mitk::DataStorage* storage, const mitk::DataNode::Pointer node);

  NIFTKMITKEXT_EXPORT mitk::DataNode::Pointer FindFirstParentImage(const mitk::DataStorage* storage, const mitk::DataNode::Pointer node, bool lookForBinary );

  NIFTKMITKEXT_EXPORT mitk::DataNode::Pointer FindNthGreyScaleImage(const std::vector<mitk::DataNode*> &nodes, int n );

  NIFTKMITKEXT_EXPORT mitk::DataNode::Pointer FindNthBinaryImage(const std::vector<mitk::DataNode*> &nodes, int n );

  NIFTKMITKEXT_EXPORT mitk::DataNode::Pointer FindFirstGreyScaleImage(const std::vector<mitk::DataNode*> &nodes );

  NIFTKMITKEXT_EXPORT mitk::DataNode::Pointer FindFirstBinaryImage(const std::vector<mitk::DataNode*> &nodes );

  NIFTKMITKEXT_EXPORT mitk::DataNode::Pointer FindNthImage(const std::vector<mitk::DataNode*> &nodes, int n, bool lookForBinary );

  NIFTKMITKEXT_EXPORT mitk::DataStorage::SetOfObjects::Pointer FindDerivedImages(const mitk::DataStorage* storage, const mitk::DataNode::Pointer node, bool lookForBinary );

  NIFTKMITKEXT_EXPORT mitk::DataStorage::SetOfObjects::Pointer FindDerivedVisibleNonHelperChildren(const mitk::DataStorage* storage, const mitk::DataNode::Pointer node);

  /**
   * \brief GetPreferedGeometry will return the geometry to use by picking one from the list of nodes, or NULL, if none can be found.
   *
   * \param nodes A vector of mitk::DataNode pointers where we assume each node has a Geometry (which should always be the case).
   * \param nodeIndex if we specify a node number/index that is a valid node in the vector <code>nodes</code> we just short-cut the search and pick that one.
   * \return mitk::TimeSlicedGeometry::Pointer A pointer to the chosen TimeSlicedGeometry or NULL if we didn't find one.
   *
   * The algorithm is:
   * <pre>
   * If nodeIndex < 0,
   *   use the first Image geometry in the list, or failing that, the first available geometry.
   * else if nodeIndex is a valid node index
   *   pick that geometry regardless of what it belongs to
   * else (user specified a useless index)
   *   pick the first available geometry.
   *
   * If the node we found was not a grey-scale image
   *   Try to find a parent grey-scale image, and if successful return that geometry.
   * </pre>
   */
  NIFTKMITKEXT_EXPORT mitk::TimeSlicedGeometry::Pointer GetPreferredGeometry(const mitk::DataStorage* dataStorage, const std::vector<mitk::DataNode*>& nodes, const int& nodeIndex=-1);
}

#endif // MITKDATASTORAGEUTILS_H

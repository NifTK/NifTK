/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkDataStorageUtils_h
#define mitkDataStorageUtils_h

#include "niftkCoreExports.h"
#include <mitkDataNode.h>
#include <mitkDataStorage.h>
#include <mitkTimeGeometry.h>
#include <mitkMIDASEnums.h>
#include <mitkMIDASImageUtils.h>

/**
 * \file mitkDataStorageUtils.h
 * \brief File containing basic DataStorage utilities such as searches, useful for a wide variety of purposes.
 */
namespace mitk
{
  NIFTKCORE_EXPORT bool IsNodeAGreyScaleImage(const mitk::DataNode::Pointer node);

  NIFTKCORE_EXPORT bool IsNodeABinaryImage(const mitk::DataNode::Pointer node);

  NIFTKCORE_EXPORT bool IsNodeAHelperObject(const mitk::DataNode* node);

  NIFTKCORE_EXPORT mitk::DataNode::Pointer FindFirstParent(const mitk::DataStorage* storage, const mitk::DataNode::Pointer node);

  NIFTKCORE_EXPORT mitk::DataNode::Pointer FindParentGreyScaleImage(const mitk::DataStorage* storage, const mitk::DataNode::Pointer node);

  NIFTKCORE_EXPORT mitk::DataNode::Pointer FindFirstParentImage(const mitk::DataStorage* storage, const mitk::DataNode::Pointer node, bool lookForBinary );

  NIFTKCORE_EXPORT mitk::DataNode::Pointer FindNthGreyScaleImage(const std::vector<mitk::DataNode*> &nodes, int n );

  NIFTKCORE_EXPORT mitk::DataNode::Pointer FindNthBinaryImage(const std::vector<mitk::DataNode*> &nodes, int n );

  NIFTKCORE_EXPORT mitk::DataNode::Pointer FindFirstGreyScaleImage(const std::vector<mitk::DataNode*> &nodes );

  NIFTKCORE_EXPORT mitk::DataNode::Pointer FindFirstBinaryImage(const std::vector<mitk::DataNode*> &nodes );

  NIFTKCORE_EXPORT mitk::DataNode::Pointer FindNthImage(const std::vector<mitk::DataNode*> &nodes, int n, bool lookForBinary );

  NIFTKCORE_EXPORT mitk::DataStorage::SetOfObjects::Pointer FindDerivedImages(const mitk::DataStorage* storage, const mitk::DataNode::Pointer node, bool lookForBinary );

  NIFTKCORE_EXPORT mitk::DataStorage::SetOfObjects::Pointer FindDerivedVisibleNonHelperChildren(const mitk::DataStorage* storage, const mitk::DataNode::Pointer node);

  /**
   * \brief GetPreferedGeometry will return the geometry to use by picking one from the list of nodes, or NULL, if none can be found.
   *
   * \param nodes A vector of mitk::DataNode pointers where we assume each node has a Geometry (which should always be the case).
   * \param nodeIndex if we specify a node number/index that is a valid node in the vector <code>nodes</code> we just short-cut the search and pick that one.
   * \return mitk::TimeGeometry::Pointer A pointer to the chosen TimeGeometry or NULL if we didn't find one.
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
  NIFTKCORE_EXPORT mitk::TimeGeometry::Pointer GetPreferredGeometry(const mitk::DataStorage* dataStorage, const std::vector<mitk::DataNode*>& nodes, const int& nodeIndex=-1);

  /**
   * \brief Loads a 4x4 matrix from a plain textfile, and puts in data storage with the given nodeName, or else creates Identity matrix.
   * \param fileName full file name
   * \param helperObject if true the node is created in DataStorage as a helper object, and so by default will normally be invisible
   */
  NIFTKCORE_EXPORT void LoadMatrixOrCreateDefault(const std::string& fileName, const std::string& nodeName, const bool& helperObject, mitk::DataStorage* dataStorage);


  /**
   * \brief Applies the given transformation to the given node.
   * \param[In] node a data node, and as each node has a mitk::Geometry3D in the mitk::BaseData, we can transform anything.
   * \param[In] transform the VTK transformation
   * \param[In] makeUndoAble if true, use the Global Undo/Redo framework, and otherwise don't.
   * \return bool true if successful and false otherwise.
   */
  NIFTKCORE_EXPORT bool ApplyToNode(mitk::DataNode::Pointer& node, const vtkMatrix4x4* transform, const bool& makeUndoAble);
}

#endif

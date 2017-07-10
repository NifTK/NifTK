/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkDataStorageUtils_h
#define niftkDataStorageUtils_h

#include "niftkCoreExports.h"

#include <mitkDataNode.h>
#include <mitkDataStorage.h>
#include <mitkTimeGeometry.h>

/// \file niftkDataStorageUtils.h
/// \brief File containing basic DataStorage utilities such as searches, useful for a wide variety of purposes.
namespace niftk
{

NIFTKCORE_EXPORT bool IsNodeABinaryImage(const mitk::DataNode* node);

NIFTKCORE_EXPORT bool IsNodeAnUcharBinaryImage(const mitk::DataNode* node);

NIFTKCORE_EXPORT bool IsNodeANonBinaryImage(const mitk::DataNode* node);

NIFTKCORE_EXPORT bool IsNodeAHelperObject(const mitk::DataNode* node);

NIFTKCORE_EXPORT mitk::DataNode* FindFirstParentImage(const mitk::DataStorage* storage, const mitk::DataNode* node, bool lookForBinary);

NIFTKCORE_EXPORT mitk::DataStorage::SetOfObjects::Pointer FindNodesStartingWith(const mitk::DataStorage* dataStorage, const std::string prefix);

/// \brief Loads a 4x4 matrix from a plain textfile, and puts in data storage with the given nodeName, or else creates Identity matrix.
/// \param fileName full file name
/// \param helperObject if true the node is created in DataStorage as a helper object, and so by default will normally be invisible
NIFTKCORE_EXPORT void LoadMatrixOrCreateDefault(const std::string& fileName, const std::string& nodeName, bool helperObject, mitk::DataStorage* dataStorage);

/// \brief Retrieves the current tranform from the mitk::Geometry object of an mitk::DataNode.
/// \param[In] node a non-NULL mitk::DataNode
/// \param[Out] outputMatrix a matrix that is updated
/// \throws mitk::Exception if node points to null object
NIFTKCORE_EXPORT void GetCurrentTransformFromNode(const mitk::DataNode::Pointer& node, vtkMatrix4x4& outputMatrix);

/// \brief Composes (pre-multiplies) the given transform with the given node.
/// \param[In] transform a VTK transformation
/// \param[Out] node a non-NULL data node, and as each node has a mitk::BaseGeometry in the mitk::BaseData, we can transform anything.
/// \throws mitk::Exception if node points to null object
NIFTKCORE_EXPORT void ComposeTransformWithNode(const vtkMatrix4x4& transform, mitk::DataNode::Pointer& node);

/// \brief Applies (sets, i.e. copies) the given transform to the given node.
/// \param[In] transform a VTK transformation
/// \param[Out] node a non-NULL data node, and as each node has a mitk::BaseGeometry in the mitk::BaseData, we can transform anything.
/// \throws mitk::Exception if node points to null object
NIFTKCORE_EXPORT void ApplyTransformToNode(const vtkMatrix4x4& transform, mitk::DataNode::Pointer& node);

}

#endif

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPolyDataUtils_h
#define niftkPolyDataUtils_h

#include "niftkCoreExports.h"

#include <mitkDataNode.h>
#include <mitkPointSet.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

namespace niftk {

/**
 * \brief Converts an mitk::PointSet to a vtkPolyData.
 */
NIFTKCORE_EXPORT void PointSetToPolyData (const mitk::PointSet::Pointer& pointsIn,
                                          vtkPolyData& polyOut);

/**
 * \brief Converts a node containing either a mitk::PointSet or mitk::Surface to a vtkPolyData.
 */
NIFTKCORE_EXPORT void NodeToPolyData (const mitk::DataNode::Pointer& node,
                                      vtkPolyData& polyOut,
                                      const mitk::DataNode::Pointer& cameraNode = mitk::DataNode::Pointer(),
                                      bool flipNormals = false);

/**
 * \brief Takes a vector of nodes, and creates a single poly data.
 */
NIFTKCORE_EXPORT vtkSmartPointer<vtkPolyData> MergePolyData(const std::vector<mitk::DataNode::Pointer>& nodes,
                                                            const mitk::DataNode::Pointer& cameraNode = mitk::DataNode::Pointer(),
                                                            bool flipNormals = false
                                                           );

} // end namespace

#endif

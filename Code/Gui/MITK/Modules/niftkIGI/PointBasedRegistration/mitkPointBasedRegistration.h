/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkPointBasedRegistration_h
#define mitkPointBasedRegistration_h

#include "niftkIGIExports.h"
#include <mitkDataStorage.h>
#include <vtkMatrix4x4.h>
#include <mitkPointSet.h>
#include <mitkDataNode.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>

namespace mitk {

/**
 * \class PointBasedRegistration
 * \brief Class to implement point based registration of two point sets.
 */
class NIFTKIGI_EXPORT PointBasedRegistration : public itk::Object
{
public:

  mitkClassMacro(PointBasedRegistration, itk::Object);
  itkNewMacro(PointBasedRegistration);

  /**
   * \brief Main method to calculate the point based registration.
   * \param[In] fixedPointSet a point set
   * \param[In] movingPointSet a point set
   * \param[In,Out] the transformation to transform the moving point set into the coordinate system of the fixed point set.
   * \return Returns the SSD of the error
   */
  double Update(const mitk::PointSet::Pointer fixedPointSet,
              const mitk::PointSet::Pointer movingPointSet,
              vtkMatrix4x4& outputTransform) const;

  /**
   * \brief Saves the given transformation to file.
   * \param[In] fileName the full absolute path of the file to be saved to, which if it already exists will be silently over-written.
   * \param[In] transform transformation matrix.
   * \return bool true if successful and false otherwise.
   */
  bool SaveToFile(const std::string& fileName, const vtkMatrix4x4& transform) const;

  /**
   * \brief Applies the given transformation to the given node.
   * \param[In] node a data node, and as each node has a mitk::Geometry3D in the mitk::BaseData, we can transfor anything.
   * \param[In] transform the VTK transformation
   * \param[In] makeUndoAble if true, use the Global Undo/Redo framework, and otherwise don't.
   * \return bool true if successful and false otherwise.
   */
  bool ApplyToNode(const mitk::DataNode::Pointer& node, vtkMatrix4x4& transform, const bool& makeUndoAble) const;

protected:

  PointBasedRegistration(); // Purposefully hidden.
  virtual ~PointBasedRegistration(); // Purposefully hidden.

  PointBasedRegistration(const PointBasedRegistration&); // Purposefully not implemented.
  PointBasedRegistration& operator=(const PointBasedRegistration&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif

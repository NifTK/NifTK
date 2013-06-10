/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkTrackedPointerManager_h
#define mitkTrackedPointerManager_h

#include "niftkIGIExports.h"
#include <vtkMatrix4x4.h>
#include <mitkDataStorage.h>
#include <mitkDataNode.h>
#include <mitkDataStorage.h>
#include <mitkPointSet.h>
#include <mitkVector.h>
#include <mitkOperation.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>

namespace mitk {

/**
 * \class TrackedPointerManager
 * \brief Command used to update the alignment of a tracked pointer.
 */
class NIFTKIGI_EXPORT TrackedPointerManager : public itk::Object,
                                              public mitk::OperationActor
{
public:

  mitkClassMacro(TrackedPointerManager, itk::Object);
  itkNewMacro(TrackedPointerManager);

  /**
   * \brief Stores the default value of whether to update the view (i.e. center it) each time the pointer moves, defaults to false.
   */
  static const bool UPDATE_VIEW_COORDINATE_DEFAULT;

  /**
   * \brief Stores the name of the mitk::PointSet that this class creates and adds to.
   */
  static const std::string TRACKED_POINTER_POINTSET_NAME;

  /**
   * \brief Used from the Undo stack, see mitk::OperationActor::ExecuteOperation().
   */
  virtual void ExecuteOperation(mitk::Operation* operation);

  /**
   * \brief Sets the data storage onto this object.
   */
  void SetDataStorage(const mitk::DataStorage::Pointer& storage);

  /**
   * \brief Takes a surface representing a tracked pointer, updates the surface's geometry, and calculates the pointer position.
   */
  void Update(
      const vtkMatrix4x4* tipToPointerTransform,
      const mitk::DataNode::Pointer pointerToWorldNode,
      mitk::DataNode::Pointer surfaceNode,
      mitk::Point3D& tipCoordinate
      );

  /**
   * \brief If the point set is not created, will create it, and then add a new point to it.
   */
  void OnGrabPoint(const mitk::Point3D& point);

  /**
   * \brief Clears the point set.
   */
  void OnClearPoints();

protected:

  TrackedPointerManager(); // Purposefully hidden.
  virtual ~TrackedPointerManager(); // Purposefully hidden.

  TrackedPointerManager(const TrackedPointerManager&); // Purposefully not implemented.
  TrackedPointerManager& operator=(const TrackedPointerManager&); // Purposefully not implemented.

private:

  /**
   * \brief Operation constant, used in Undo/Redo framework.
   */
  static const mitk::OperationType OP_UPDATE_POINTSET;

  /**
   * \brief Returns the point set from data storage, creating one if it can't be found.
   */
  mitk::PointSet::Pointer RetrievePointSet();

  /**
   * \brief Stores a local reference to the data storage.
   */
  mitk::DataStorage::Pointer m_DataStorage;

}; // end class

} // end namespace

#endif

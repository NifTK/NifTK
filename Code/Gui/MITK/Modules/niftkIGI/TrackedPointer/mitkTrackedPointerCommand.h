/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkTrackedPointerCommand_h
#define mitkTrackedPointerCommand_h

#include "niftkIGIExports.h"
#include <mitkDataStorage.h>
#include <vtkMatrix4x4.h>
#include <mitkDataNode.h>
#include <itkObject.h>
#include <itkObjectFactoryBase.h>

namespace mitk {

/**
 * \class TrackedPointerCommand
 * \brief Command used to update the alignment of a tracked pointer.
 */
class NIFTKIGI_EXPORT TrackedPointerCommand : public itk::Object
{
public:

  mitkClassMacro(TrackedPointerCommand, itk::Object);
  itkNewMacro(TrackedPointerCommand);

  static const bool UPDATE_VIEW_COORDINATE_DEFAULT;

  /**
   * \brief Takes a surface representing a tracked pointer, updates the surface's geometry, and calculates the pointer position.
   */
  void Update(
      const vtkMatrix4x4* tipToPointerTransform,
      const mitk::DataNode::Pointer pointerToWorldNode,
      mitk::DataNode::Pointer surfaceNode,
      mitk::Point3D& tipCoordinate
      );

protected:

  TrackedPointerCommand(); // Purposefully hidden.
  virtual ~TrackedPointerCommand(); // Purposefully hidden.

  TrackedPointerCommand(const TrackedPointerCommand&); // Purposefully not implemented.
  TrackedPointerCommand& operator=(const TrackedPointerCommand&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif

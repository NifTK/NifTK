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

  /**
   * \brief Write My Documentation
   */
  void Update(const mitk::DataStorage::Pointer dataStorage,
           const mitk::DataNode::Pointer imageNode,
           const mitk::DataNode::Pointer surfaceNode,
           const mitk::DataNode::Pointer probeToWorldNode,
           const vtkMatrix4x4* imageToProbeTransform);

protected:

  TrackedPointerCommand(); // Purposefully hidden.
  virtual ~TrackedPointerCommand(); // Purposefully hidden.

  TrackedPointerCommand(const TrackedPointerCommand&); // Purposefully not implemented.
  TrackedPointerCommand& operator=(const TrackedPointerCommand&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif

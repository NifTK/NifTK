/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkAuroraCubeTracker_h
#define niftkAuroraCubeTracker_h

#include <niftkNDITrackersExports.h>
#include "niftkMITKNDITracker.h"

namespace niftk
{

/**
 * \class AuroraCubeTracker
 * \brief RAII object to connect to Aurora Cube tracker.
 */
class NIFTKNDITRACKERS_EXPORT AuroraCubeTracker : public niftk::MITKNDITracker
{
public:

  mitkClassMacroItkParent(AuroraCubeTracker, niftk::MITKNDITracker);
  mitkNewMacro3Param(AuroraCubeTracker, mitk::DataStorage::Pointer, std::string, std::string);

protected:

  AuroraCubeTracker(mitk::DataStorage::Pointer dataStorage,
                    std::string portName,
                    std::string toolConfigFileName); // Purposefully hidden.

  virtual ~AuroraCubeTracker(); // Purposefully hidden.

  AuroraCubeTracker(const AuroraCubeTracker&); // Purposefully not implemented.
  AuroraCubeTracker& operator=(const AuroraCubeTracker&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif

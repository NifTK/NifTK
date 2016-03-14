/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkAuroraDomeTracker_h
#define niftkAuroraDomeTracker_h

#include <niftkIGITrackersExports.h>
#include "niftkNDITracker.h"

namespace niftk
{

/**
 * \class AuroraDomeTracker
 * \brief RAII object to connect to Aurora Dome tracker.
 */
class NIFTKIGITRACKERS_EXPORT AuroraDomeTracker : public niftk::NDITracker
{
public:

  mitkClassMacroItkParent(AuroraDomeTracker, niftk::NDITracker);
  mitkNewMacro3Param(AuroraDomeTracker, mitk::DataStorage::Pointer, std::string, std::string);

protected:

  AuroraDomeTracker(mitk::DataStorage::Pointer dataStorage,
                    std::string portName,
                    std::string toolConfigFileName); // Purposefully hidden.

  virtual ~AuroraDomeTracker(); // Purposefully hidden.

  AuroraDomeTracker(const AuroraDomeTracker&); // Purposefully not implemented.
  AuroraDomeTracker& operator=(const AuroraDomeTracker&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif

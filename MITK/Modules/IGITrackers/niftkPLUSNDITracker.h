/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPLUSNDITracker_h
#define niftkPLUSNDITracker_h

#include "niftkNDITracker.h"
#include <niftkIGITrackersExports.h>

namespace niftk {

/**
* \class PLUSNDITracker
* \brief RAII wrapper for PLUS/Atami interface to NDI trackers.
*/
class NIFTKIGITRACKERS_EXPORT PLUSNDITracker : public niftk::NDITracker
{
public:

  mitkClassMacroItkParent(PLUSNDITracker, niftk::NDITracker);

protected:

  PLUSNDITracker(mitk::DataStorage::Pointer dataStorage,
                 std::string portName,
                 mitk::TrackingDeviceData deviceData,
                 std::string toolConfigFileName,
                 int preferredFramesPerSecond); // Purposefully hidden.

  virtual ~PLUSNDITracker(); // Purposefully hidden.

  PLUSNDITracker(const PLUSNDITracker&); // Purposefully not implemented.
  PLUSNDITracker& operator=(const PLUSNDITracker&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif


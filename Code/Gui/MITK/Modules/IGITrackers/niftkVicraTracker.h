/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkVicraTracker_h
#define niftkVicraTracker_h

#include <niftkIGITrackersExports.h>
#include "niftkNDITracker.h"

namespace niftk
{

/**
 * \class VicraTracker
 * \brief RAII object to connect to Polaris Vicra tracker.
 */
class NIFTKIGITRACKERS_EXPORT VicraTracker : public niftk::NDITracker
{
public:

  mitkClassMacroItkParent(VicraTracker, niftk::NDITracker);
  mitkNewMacro3Param(VicraTracker, mitk::DataStorage::Pointer, mitk::SerialCommunication::PortNumber, std::string);

protected:

  VicraTracker(mitk::DataStorage::Pointer dataStorage,
               mitk::SerialCommunication::PortNumber portNumber,
               std::string toolConfigFileName); // Purposefully hidden.

  virtual ~VicraTracker(); // Purposefully hidden.

  VicraTracker(const VicraTracker&); // Purposefully not implemented.
  VicraTracker& operator=(const VicraTracker&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif

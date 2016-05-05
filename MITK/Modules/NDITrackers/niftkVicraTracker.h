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

#include <niftkNDITrackersExports.h>
#include "niftkPLUSNDITracker.h"

namespace niftk
{

/**
 * \class VicraTracker
 * \brief RAII object to connect to Polaris Vicra tracker.
 */
class NIFTKNDITRACKERS_EXPORT VicraTracker : public niftk::PLUSNDITracker
{
public:

  mitkClassMacroItkParent(VicraTracker, niftk::PLUSNDITracker);
  mitkNewMacro4Param(VicraTracker, mitk::DataStorage::Pointer, std::string, std::string, int);

protected:

  VicraTracker(mitk::DataStorage::Pointer dataStorage,
               std::string portName,
               std::string toolConfigFileName,
               int baudRate
               ); // Purposefully hidden.

  virtual ~VicraTracker(); // Purposefully hidden.

  VicraTracker(const VicraTracker&); // Purposefully not implemented.
  VicraTracker& operator=(const VicraTracker&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif

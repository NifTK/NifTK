/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkSpectraTracker_h
#define niftkSpectraTracker_h

#include <niftkNDITrackersExports.h>
#include "niftkPLUSNDITracker.h"

namespace niftk
{

/**
 * \class SpectraTracker
 * \brief RAII object to connect to Polaris Spectra tracker.
 */
class NIFTKNDITRACKERS_EXPORT SpectraTracker : public niftk::PLUSNDITracker
{
public:

  mitkClassMacroItkParent(SpectraTracker, niftk::PLUSNDITracker);
  mitkNewMacro4Param(SpectraTracker, mitk::DataStorage::Pointer, std::string, std::string, int);

protected:

  SpectraTracker(mitk::DataStorage::Pointer dataStorage,
                 std::string portName,
                 std::string toolConfigFileName,
                 int baudRate
                 ); // Purposefully hidden.

  virtual ~SpectraTracker(); // Purposefully hidden.

  SpectraTracker(const SpectraTracker&); // Purposefully not implemented.
  SpectraTracker& operator=(const SpectraTracker&); // Purposefully not implemented.

private:

}; // end class

} // end namespace

#endif

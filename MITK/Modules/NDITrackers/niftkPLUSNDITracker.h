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
#include <niftkNDITrackersExports.h>
#include <niftkNDICAPITracker.h>

namespace niftk {

/**
* \class PLUSNDITracker
* \brief RAII wrapper for PLUS/Atami interface to NDI trackers.
*/
class NIFTKNDITRACKERS_EXPORT PLUSNDITracker : public niftk::NDITracker
{
public:

  mitkClassMacroItkParent(PLUSNDITracker, niftk::NDITracker)

  /**
  * \brief Retrives the current tracking data.
  */
  virtual std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > GetTrackingData() override;

protected:

  PLUSNDITracker(mitk::DataStorage::Pointer dataStorage,
                 std::string portName, // this should ALWAYS be an int.
                                       // but we store it as a string to pass to base class.
                 mitk::TrackingDeviceData deviceData,
                 std::string toolConfigFileName,
                 int preferredFramesPerSecond,
                 int baudRate,
                 int measurementVolumeNumber
                 ); // Purposefully hidden.

  virtual ~PLUSNDITracker(); // Purposefully hidden.

  PLUSNDITracker(const PLUSNDITracker&); // Purposefully not implemented.
  PLUSNDITracker& operator=(const PLUSNDITracker&); // Purposefully not implemented.

  /**
  * \brief converts the name=/dev/cu.bluetooth (or similar), to an index in the list of enumerated ports.
  */
  std::string ConvertPortNameToPortIndexPlusOne(const std::string& name) const;

private:

  niftk::NDICAPITracker m_Tracker;

  unsigned int m_SuppressUpateErrorsAfterNRepeats;  // To stop repeated calls to internal update flooding the console.
  unsigned int m_UpdateErrorRepeatCounter;          // To stop repeated calls to internal update flooding the console.

}; // end class

} // end namespace

#endif


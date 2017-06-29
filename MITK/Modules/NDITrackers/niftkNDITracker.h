/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNDITracker_h
#define niftkNDITracker_h

#include <niftkNDITrackersExports.h>
#include <niftkIGITracker.h>
#include <mitkTrackingVolumeGenerator.h>

namespace niftk
{

/**
 * \class NDITracker
 * \brief Abstract base class for NifTK interfaces to NDI trackers.
 * \see IGITrackers
 */
class NIFTKNDITRACKERS_EXPORT NDITracker : public niftk::IGITracker
{
public:

  mitkClassMacroItkParent(NDITracker, niftk::IGITracker)

  /**
  * \see niftk::IGITracker::GetTrackingData()
  */
  virtual std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > GetTrackingData() = 0;

protected:

  NDITracker(mitk::DataStorage::Pointer dataStorage,
             std::string portName,
             mitk::TrackingDeviceData deviceData,
             std::string toolConfigFileName,
             int preferredFramesPerSecond); // Purposefully hidden.

  virtual ~NDITracker(); // Purposefully hidden.

  NDITracker(const NDITracker&); // Purposefully not implemented.
  NDITracker& operator=(const NDITracker&); // Purposefully not implemented.

  // Passed in to constructor, and stored in this class (see also base class).
  std::string                            m_PortName;
  mitk::TrackingDeviceData               m_DeviceData;

  // Created during constructor, and stored in this class (see also base class).
  mitk::TrackingVolumeGenerator::Pointer m_TrackingVolumeGenerator;

}; // end class

} // end namespace

#endif

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkMITKNDITracker_h
#define niftkMITKNDITracker_h

#include "niftkNDITracker.h"
#include <niftkIGITrackersExports.h>
#include <mitkTrackingDeviceSource.h>
#include <mitkNDITrackingDevice.h>

namespace niftk
{

/**
 * \class MITKNDITracker
 * \brief RAII wrapper for MITK interface to NDI trackers.
 */
class NIFTKIGITRACKERS_EXPORT MITKNDITracker : public niftk::NDITracker
{
public:

  mitkClassMacroItkParent(MITKNDITracker, niftk::NDITracker);

  /**
  * \brief Retrives the current tracking data.
  */
  virtual std::map<std::string, vtkSmartPointer<vtkMatrix4x4> > GetTrackingData();

protected:

  MITKNDITracker(mitk::DataStorage::Pointer dataStorage,
                 std::string portName,
                 mitk::TrackingDeviceData deviceData,
                 std::string toolConfigFileName,
                 int preferredFramesPerSecond); // Purposefully hidden.

  virtual ~MITKNDITracker(); // Purposefully hidden.

  MITKNDITracker(const MITKNDITracker&); // Purposefully not implemented.
  MITKNDITracker& operator=(const MITKNDITracker&); // Purposefully not implemented.

  void StartTracking();
  void StopTracking();
  bool IsTracking() const;
  void OpenConnection();
  void CloseConnection();

private:

  // Created during constructor.
  mitk::NDITrackingDevice::Pointer    m_TrackerDevice;
  mitk::TrackingDeviceSource::Pointer m_TrackerSource;

}; // end class

} // end namespace

#endif

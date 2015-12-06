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

#include <niftkIGITrackersExports.h>
#include <niftkNDITracker.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkDataStorage.h>
#include <mitkNavigationToolStorage.h>
#include <mitkTrackingDeviceSource.h>
#include <mitkNDITrackingDevice.h>
#include <mitkSerialCommunication.h>
#include <mitkTrackingVolumeGenerator.h>
#include <mitkTrackingTypes.h>
#include <map>

namespace niftk
{

/**
 * \class NDITracker
 * \brief RAII Helper object to wrap classes and standardise access to our NDI trackers.
 */
class NIFTKIGITRACKERS_EXPORT NDITracker : public itk::Object
{
public:

  mitkClassMacroItkParent(NDITracker, itk::Object);
//  mitkNewMacro5Param(NDITracker, mitk::DataStorage::Pointer, mitk::SerialCommunication::PortNumber, mitk::TrackingDeviceType, mitk::TrackingDeviceData, std::string);

  itkGetMacro(PreferredFramesPerSecond, int);

  /**
  * \brief Starts the tracking.
  *
  * Note that the constructor should have already connected.
  */
  void StartTracking();

  /**
  * \brief Stops tracking.
  *
  * Note that you can start/stop, but the device should always be connected.
  */
  void StopTracking();

  /**
  * \brief Set the tracking volume visible or invisible.
  *
  * Each tracker loads a tracking volume for visualisation purposes.
  */
  void SetVisibilityOfTrackingVolume(bool isVisible);

  /**
  * \brief Get the visibility flag for the tracking volume.
  */
  bool GetVisibilityOfTrackingVolume() const;

  /**
  * \brief Updates the pipeline, meaning, it retrieves the tracking data from the device.
  */
  void Update();

  /**
  * \brief Retrives the current tracking data.
  */
  std::map<std::string, vtkSmartPointer<vtkMatrix4x4> > GetTrackingData();

protected:

  NDITracker(mitk::DataStorage::Pointer dataStorage,
             mitk::SerialCommunication::PortNumber portNumber,
             mitk::TrackingDeviceType deviceType,
             mitk::TrackingDeviceData deviceData,
             std::string toolConfigFileName,
             int preferredFramesPerSecond); // Purposefully hidden.

  virtual ~NDITracker(); // Purposefully hidden.

  NDITracker(const NDITracker&); // Purposefully not implemented.
  NDITracker& operator=(const NDITracker&); // Purposefully not implemented.

private:

  void OpenConnection();
  void CloseConnection();

  // Passed in via constructor.
  mitk::DataStorage::Pointer               m_DataStorage;
  mitk::SerialCommunication::PortNumber    m_PortNumber;
  mitk::TrackingDeviceType                 m_DeviceType;
  mitk::TrackingDeviceData                 m_DeviceData;
  std::string                              m_ToolConfigFileName;
  int                                      m_PreferredFramesPerSecond;

  // Created during constructor.
  mitk::NavigationToolStorage::Pointer     m_NavigationToolStorage;
  mitk::NDITrackingDevice::Pointer         m_TrackerDevice;
  mitk::TrackingDeviceSource::Pointer      m_TrackerSource;
  mitk::TrackingVolumeGenerator::Pointer   m_TrackingVolumeGenerator;
  mitk::DataNode::Pointer                  m_TrackingVolumeNode;

}; // end class

} // end namespace

#endif

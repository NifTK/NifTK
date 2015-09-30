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
#include <mitkNDITrackingDevice.h>
#include <mitkSerialCommunication.h>
#include <mitkTrackingVolumeGenerator.h>
#include <mitkTrackingTypes.h>

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
  mitkNewMacro5Param(NDITracker, mitk::DataStorage::Pointer, mitk::SerialCommunication::PortNumber, mitk::TrackingDeviceType, mitk::TrackingDeviceData, std::string);

  void StartTracking();
  void StopTracking();
  void SetVisibilityOfTrackingVolume(bool isVisible);
  bool GetVisibilityOfTrackingVolume() const;

protected:

  NDITracker(mitk::DataStorage::Pointer dataStorage,
             mitk::SerialCommunication::PortNumber portNumber,
             mitk::TrackingDeviceType deviceType,
             mitk::TrackingDeviceData deviceData,
             std::string toolConfigFileName); // Purposefully hidden.

  virtual ~NDITracker(); // Purposefully hidden.

  NDITracker(const NDITracker&); // Purposefully not implemented.
  NDITracker& operator=(const NDITracker&); // Purposefully not implemented.

private:

  void OpenConnection();
  void CloseConnection();

  // Passed in via constructor.
  mitk::DataStorage::Pointer             m_DataStorage;
  mitk::SerialCommunication::PortNumber  m_PortNumber;
  mitk::TrackingDeviceType               m_DeviceType;
  mitk::TrackingDeviceData               m_DeviceData;
  std::string                            m_ToolConfigFileName;

  // Created during constructor.
  mitk::NavigationToolStorage::Pointer   m_NavigationToolStorage;
  mitk::NDITrackingDevice::Pointer       m_TrackerDevice;
  mitk::TrackingVolumeGenerator::Pointer m_TrackingVolumeGenerator;
  mitk::DataNode::Pointer                m_TrackingVolumeNode;

}; // end class

} // end namespace

#endif

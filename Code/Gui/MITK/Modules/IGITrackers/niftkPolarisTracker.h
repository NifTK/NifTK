/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkPolarisTracker_h
#define niftkPolarisTracker_h

#include <niftkIGITrackersExports.h>

#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkDataStorage.h>
#include <mitkNavigationToolStorage.h>
#include <mitkNDITrackingDevice.h>
#include <mitkSerialCommunication.h>
#include <mitkTrackingVolumeGenerator.h>

namespace niftk
{

/**
 * \class PolarisTracker
 * \brief RAII object to connect to Polaris (Vicra, Spectra) tracker.
 */
class NIFTKIGITRACKERS_EXPORT PolarisTracker : public itk::Object
{
public:

  mitkClassMacroItkParent(PolarisTracker, itk::Object);
  mitkNewMacro3Param(PolarisTracker, mitk::DataStorage::Pointer, mitk::SerialCommunication::PortNumber, std::string);

  void StartTracking();
  void StopTracking();
  void SetVisibilityOfTrackingVolume(bool isVisible);

  itkGetMacro(PreferredFramesPerSecond, int);

protected:

  PolarisTracker(mitk::DataStorage::Pointer dataStorage,
                 mitk::SerialCommunication::PortNumber portNumber,
                 std::string toolConfigFileName); // Purposefully hidden.

  virtual ~PolarisTracker(); // Purposefully hidden.

  PolarisTracker(const PolarisTracker&); // Purposefully not implemented.
  PolarisTracker& operator=(const PolarisTracker&); // Purposefully not implemented.

private:

  void OpenConnection(); // Purposefully private.
  void CloseConnection(); // Purposefully private.

  int                                    m_PreferredFramesPerSecond;
  mitk::DataStorage::Pointer             m_DataStorage;
  mitk::SerialCommunication::PortNumber  m_PortNumber;
  std::string                            m_ToolConfigFileName;
  std::string                            m_VolumeOfInterestFileName;
  mitk::NavigationToolStorage::Pointer   m_NavigationToolStorage;
  mitk::NDITrackingDevice::Pointer       m_TrackerDevice;
  mitk::TrackingVolumeGenerator::Pointer m_TrackingVolumeGenerator;
  mitk::DataNode::Pointer                m_TrackingVolumeNode;

}; // end class

} // end namespace

#endif

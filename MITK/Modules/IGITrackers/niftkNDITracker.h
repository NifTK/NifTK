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
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkDataStorage.h>
#include <mitkNavigationToolStorage.h>
#include <mitkTrackingVolumeGenerator.h>
#include <map>

namespace niftk
{

/**
 * \class NDITracker
 * \brief Base class for NifTK interfaces to NDI trackers.
 */
class NIFTKIGITRACKERS_EXPORT NDITracker : public itk::Object
{
public:

  mitkClassMacroItkParent(NDITracker, itk::Object);
  itkGetMacro(PreferredFramesPerSecond, int);

  /**
  * \brief Retrives the current tracking data.
  */
  virtual std::map<std::string, vtkSmartPointer<vtkMatrix4x4> > GetTrackingData() = 0;

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

protected:

  NDITracker(mitk::DataStorage::Pointer dataStorage,
             std::string portName,
             mitk::TrackingDeviceData deviceData,
             std::string toolConfigFileName,
             int preferredFramesPerSecond); // Purposefully hidden.

  virtual ~NDITracker(); // Purposefully hidden.

  NDITracker(const NDITracker&); // Purposefully not implemented.
  NDITracker& operator=(const NDITracker&); // Purposefully not implemented.

  // Passed in via constructor.
  mitk::DataStorage::Pointer               m_DataStorage;
  std::string                              m_PortName;
  mitk::TrackingDeviceData                 m_DeviceData;
  std::string                              m_ToolConfigFileName;
  int                                      m_PreferredFramesPerSecond;

  // Created during constructor.
  mitk::NavigationToolStorage::Pointer     m_NavigationToolStorage;
  mitk::TrackingVolumeGenerator::Pointer   m_TrackingVolumeGenerator;
  mitk::DataNode::Pointer                  m_TrackingVolumeNode;

}; // end class

} // end namespace

#endif

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
 *
 * The main design point is that tracking starts during the
 * constructor, and stops during the destructor. All errors
 * are thrown as mitk::Exception. Then you repeatedly
 * call GetTrackingData(). This class does not do
 * threading. It is assumed that the calling client
 * can manage buffers of matrices, threading etc.
 * This thing just grabs the latest tracking matrices each time
 * you you call GetTrackingData().
 */
class NIFTKNDITRACKERS_EXPORT NDITracker : public itk::Object
{
public:

  mitkClassMacroItkParent(NDITracker, itk::Object)
  itkGetMacro(PreferredFramesPerSecond, int);

  /**
  * \brief Retrives the current tracking data.
  * \return map of tool-name and tracking matrix.
  *
  * Given that errors are thrown during construction,
  * (like RAII pattern), then once the constructor
  * is complete we can assume that the tracker is valid.
  *
  * Therefore we repeated call this method. If a tool is
  * not visible, nothing for that tool will be returned.
  * So, this will return a varying number of items, depending
  * on how many tools are visible to the tracker. So
  * zero items returned is a valid output.
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

  // Passed in to constructor.
  mitk::DataStorage::Pointer             m_DataStorage;
  std::string                            m_PortName;
  mitk::TrackingDeviceData               m_DeviceData;
  std::string                            m_ToolConfigFileName;
  int                                    m_PreferredFramesPerSecond;

  // Created during constructor.
  mitk::NavigationToolStorage::Pointer   m_NavigationToolStorage;
  mitk::TrackingVolumeGenerator::Pointer m_TrackingVolumeGenerator;
  mitk::DataNode::Pointer                m_TrackingVolumeNode;

}; // end class

} // end namespace

#endif

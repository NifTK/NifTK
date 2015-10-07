/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNDIBaseTracker_h
#define niftkNDIBaseTracker_h

#include <niftkIGITrackersExports.h>
#include "niftkNDITracker.h"
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>

namespace niftk
{

/**
* \class NDIBaseTracker
* \brief Base class, RAII object, to connect to NDI Trackers.
*/
class NIFTKIGITRACKERS_EXPORT NDIBaseTracker : public itk::Object
{
public:

  mitkClassMacroItkParent(NDIBaseTracker, itk::Object);
  itkNewMacro(NDIBaseTracker);

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
  * \brief Calibrate the time lag, and set it here, in milliseconds.
  */
  void SetDelayInMilliseconds(unsigned int);

  /**
  * \brief Updates the pipeline, meaning, it retrieves the tracking data from the device.
  */
  void Update();

  /**
  * \brief Retrives the current tracking data, taking into account any calibrated lag.
  */
  std::map<std::string, vtkSmartPointer<vtkMatrix4x4> > GetTrackingData();

protected:

  NDIBaseTracker(); // Purposefully hidden.
  virtual ~NDIBaseTracker(); // Purposefully hidden.

  NDIBaseTracker(const NDIBaseTracker&); // Purposefully not implemented.
  NDIBaseTracker& operator=(const NDIBaseTracker&); // Purposefully not implemented.

  int                        m_PreferredFramesPerSecond;
  niftk::NDITracker::Pointer m_Tracker;

}; // end class

} // end namespace

#endif

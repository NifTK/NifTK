/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkAtracsysTracker_h
#define niftkAtracsysTracker_h

#include "niftkAtracsysExports.h"
#include <niftkIGITracker.h>
#include <memory>

namespace niftk
{

class AtracsysTrackerPrivate;

/**
 * \class AtracsysTracker
 * \brief Interface to Atracsys tracker that can be used
 * from either command line or GUI clients within NifTK.
 *
 * The Atracsys API provides a blocking call, that will wait
 * until the next frame is available. Therefore, this class
 * contains no threading. You are expected to instantiate this
 * class, and then poll it at your desired rate.
 */
class NIFTKATRACSYS_EXPORT AtracsysTracker : public niftk::IGITracker
{

public:

  mitkClassMacroItkParent(AtracsysTracker, niftk::IGITracker)
  mitkNewMacro2Param(AtracsysTracker, mitk::DataStorage::Pointer, std::string)

  /**
  * \see niftk::IGITracker::GetTrackingData()
  */
  virtual std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > GetTrackingData();

  /**
  * \brief Experimental research interface, returns each ball position.
  */
  std::vector<mitk::Point3D> GetBallPositions();

  /**
  * \brief Even more experimental, returns markers and any spare balls from the same frame.
  */
  void GetMarkersAndBalls(std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> >&,
                          std::vector<mitk::Point3D>&
                         );

protected:

  AtracsysTracker(mitk::DataStorage::Pointer dataStorage,
                  std::string toolConfigFileName
                 ); // Purposefully hidden.

  virtual ~AtracsysTracker(); // Purposefully hidden.

  AtracsysTracker(const AtracsysTracker&); // Purposefully not implemented.
  AtracsysTracker& operator=(const AtracsysTracker&); // Purposefully not implemented.

private:

  std::unique_ptr<AtracsysTrackerPrivate> m_Tracker; // PIMPL pattern.

}; // end class

} // end namespace

#endif

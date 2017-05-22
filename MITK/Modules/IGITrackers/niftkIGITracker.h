/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkIGITracker_h
#define niftkIGITracker_h

#include <niftkIGITrackersExports.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include <mitkVector.h>
#include <mitkDataStorage.h>
#include <mitkNavigationToolStorage.h>
#include <map>

namespace niftk
{

/**
 * \class IGITracker
 * \brief Abstract base class for NifTK interfaces to any tracker device.
 *
 * The main design point for derived classes are:
 *
 * <verbatim>
 *   1. Tracking starts in constructor.
 *   2. Tracking stops in destructor.
 *   3. All errors thrown as mitk::Exception or subclasses of mitk::Exception.
 *   4. These classes can NOT be assumed to be thread safe.
 *   5. Then you preferably call GetTrackingData() or less-preferably GetTrackingDataAsMatrix().
 *   6. If an object is untracked/invisible, no entry is returned, so GetTrackingData
 *      and GetTrackingDataAsMatrix() can return an empty list.
 * </verbatim>
 *
 * Each class should, but doesn't have to, provide an mitk::Surface describing
 * the tracking volume. This is really useful for visualisation, as it sets up
 * the right size of the 3D region of interest.
 */
class NIFTKIGITRACKERS_EXPORT IGITracker : public itk::Object
{
public:

  mitkClassMacroItkParent(IGITracker, itk::Object)
  itkGetMacro(PreferredFramesPerSecond, int);

  /**
  * \brief Returns the current tracking data as a translation vector and quaternion.
  * \brief map of tool-name and quaternion.
  */
  virtual std::map<std::string, std::pair<mitk::Point4D, mitk::Vector3D> > GetTrackingData() = 0;

  /**
  * \brief Retrives the current tracking data as matrix.
  * \return map of tool-name and tracking matrix.
  */
  std::map<std::string, vtkSmartPointer<vtkMatrix4x4> > GetTrackingDataAsMatrices();

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

  IGITracker(mitk::DataStorage::Pointer dataStorage,
             std::string toolConfigFileName,
             int preferredFramesPerSecond); // Purposefully hidden.

  virtual ~IGITracker(); // Purposefully hidden.

  IGITracker(const IGITracker&); // Purposefully not implemented.
  IGITracker& operator=(const IGITracker&); // Purposefully not implemented.

  // Passed in to constructor.
  mitk::DataStorage::Pointer             m_DataStorage;
  std::string                            m_ToolConfigFileName;
  int                                    m_PreferredFramesPerSecond;

  // Created during constructor.
  mitk::NavigationToolStorage::Pointer   m_NavigationToolStorage;
  mitk::DataNode::Pointer                m_TrackingVolumeNode;

}; // end class

} // end namespace

#endif

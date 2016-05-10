/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkNiftyCalVideoCalibrateManager_h
#define niftkNiftyCalVideoCalibrateManager_h

#include "niftkNiftyCalExports.h"
#include <itkObject.h>
#include <itkObjectFactoryBase.h>
#include <mitkDataStorage.h>
#include <mitkDataNode.h>
#include <list>
#include <niftkPointUtilities.h>

namespace niftk {

/**
 * \class NiftyCalVideoCalibrationManager
 * \brief Manager class to perform video calibration as provided by NiftyCal.
 *
 * This one is not an MITK Service as it is stateful. So, it would
 * be unwise to have a system-wide service, called from multiple threads.
 */
class NIFTKNIFTYCAL_EXPORT NiftyCalVideoCalibrationManager : public itk::Object
{
public:

  mitkClassMacroItkParent(NiftyCalVideoCalibrationManager, itk::Object);
  itkNewMacro(NiftyCalVideoCalibrationManager);

  void SetDataStorage(const mitk::DataStorage::Pointer& storage);

  itkSetMacro(LeftImageNode, mitk::DataNode::Pointer);
  itkGetMacro(LeftImageNode, mitk::DataNode::Pointer);

  itkSetMacro(RightImageNode, mitk::DataNode::Pointer);
  itkGetMacro(RightImageNode, mitk::DataNode::Pointer);

  itkSetMacro(TrackingTransformNode, mitk::DataNode::Pointer);
  itkGetMacro(TrackingTransformNode, mitk::DataNode::Pointer);

  itkSetMacro(MinimumNumberOfSnapshotsForCalibrating, unsigned int);
  itkGetMacro(MinimumNumberOfSnapshotsForCalibrating, unsigned int);

  unsigned int GetNumberOfSnapshots() const;

  /**
   * \brief Clears down the internal points and image arrays,
   * so calibration will restart with zero data.
   */
  void Restart();

  /**
   * \brief Grabs images and tracking, and runs the point extraction.
   * \return Returns true if successful and false otherwise.
   */
  bool Grab();

  /**
   * \brief Removes the last grabbed snapshot.
   */
  void UnGrab();

  /**
   * \brief Performs the actual calibration.
   *
   * This can be mono, stereo, iterative and include hand-eye,
   * depending on the configuration parameters stored in this class.
   *
   * \return rms re-projection error (pixels)
   */
  double Calibrate();

  /**
   * \brief Saves a bunch of standard (from a NifTK perspective)
   * calibration files to the given directory path. If files
   * exist, they will be moved and given a timestamp of when
   * they were moved.
   */
  void Save(const std::string dirName);

protected:

  NiftyCalVideoCalibrationManager(); // Purposefully hidden.
  virtual ~NiftyCalVideoCalibrationManager(); // Purposefully hidden.

  NiftyCalVideoCalibrationManager(const NiftyCalVideoCalibrationManager&); // Purposefully not implemented.
  NiftyCalVideoCalibrationManager& operator=(const NiftyCalVideoCalibrationManager&); // Purposefully not implemented.

private:

  mitk::DataStorage::Pointer m_DataStorage;
  mitk::DataNode::Pointer    m_LeftImageNode;
  mitk::DataNode::Pointer    m_RightImageNode;
  mitk::DataNode::Pointer    m_TrackingTransformNode;
  unsigned int               m_MinimumNumberOfSnapshotsForCalibrating;

  // Data used for calibration.
  std::list<niftk::PointSet> m_LeftPoints;
  std::list<niftk::PointSet> m_RightPoints;

}; // end class

} // end namespace

#endif

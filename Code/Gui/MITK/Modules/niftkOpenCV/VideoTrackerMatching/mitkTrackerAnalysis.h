/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkTrackerAnalysis_h
#define mitkTrackerAnalysis_h

#include "niftkOpenCVExports.h"
#include "mitkVideoTrackerMatching.h"
#include <cv.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

namespace mitk
{


/**
 * \brief A class to match video frames to tracking frames, when reading 
 * recorded tracking data. 
 */
class NIFTKOPENCV_EXPORT TrackerAnalysis : public VideoTrackerMatching
{
public: 
  mitkClassMacro ( TrackerAnalysis, itk::Object);
  itkNewMacro (TrackerAnalysis);

  /**
   * \brief Pass a file name that defines the position of a point fixed in world
   * coordinates relative to the camera lens, along with the on screen coordinates
   * of the point. The VideoLag is adjusted so as to 
   * minimalise the standard deviation of the reconstructed world point
   */
  void TemporalCalibration (std::string filename, int windowLow = -100, int windowHigh = 100, bool visualise = false , std::string fileout = "" );

  /**
   * \brief Pass a file name that defines the position of a point fixed in world
   * coordinates relative to the camera lens, along with the on screen coordinates 
   * of the point. The world position of the point and 
   * the handeye calibration are optimised to minimise the residual error of the 
   * reconstructed point
   */
  void OptimiseHandeyeCalibration (std::string filename, bool visualise = false , std::string fileout = "" );

  /**
   * \brief Pass a file name that defines the position of a point fixed in world
   * coordinates relative to the camera lens, along with the on screen coordinates of 
   * the point. The world position of the point is 
   * is determined using a range of perturbed values for the handeye matrix. The
   * variance in the residual reconstruction error is used to determine the 
   * sensitivity of the system to errors in the hand eye calibration
   */
  void HandeyeSensitivityTest (std::string filename, bool visualise = false , std::string fileout = "" );

protected:
  TrackerAnalysis();
  virtual ~TrackerAnalysis();

  TrackerAnalysis(const TrackerAnalysis&); // Purposefully not implemented.
  TrackerAnalysis& operator=(const TrackerAnalysis&); // Purposefully not implemented.

private:
  std::string m_CalibrationDirectory;    //the directory containing camera intrinsic parameters  
};


} // namespace


#endif // niftkTrackerAnalysis_h

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkTwoTrackerAnalysis_h
#define mitkTwoTrackerAnalysis_h

#include "niftkOpenCVExports.h"
#include "mitkTwoTrackerMatching.h"
#include <cv.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

namespace mitk
{


/**
 * \brief A class to match video frames to tracking frames, or two
 * sets of tracking data when reading 
 * recorded tracking data. 
 */
class NIFTKOPENCV_EXPORT TwoTrackerAnalysis : public TwoTrackerMatching
{
public: 
  mitkClassMacro ( TwoTrackerAnalysis, itk::Object);
  itkNewMacro (TwoTrackerAnalysis);

  /**
   * \brief Temporal calibration of two tracking streams. The Lag is adjusted so as to 
   * minimalise maximise the correlation between the tracking signals
   */
  void TemporalCalibration (int windowLow = -100, int windowHigh = 100, bool visualise = false , std::string fileout = "" );

  /**
   * \brief Pass a file name that defines the position of a point fixed in world
   * coordinates relative to the camera lens, along with the on screen coordinates 
   * of the point. The world position of the point and 
   * the handeye calibration are optimised to minimise the residual error of the 
   * reconstructed point
   */
  void HandeyeCalibration (bool visualise = false , std::string fileout = "" );

protected:
  TwoTrackerAnalysis();
  virtual ~TwoTrackerAnalysis();

  TwoTrackerAnalysis(const TwoTrackerAnalysis&); // Purposefully not implemented.
  TwoTrackerAnalysis& operator=(const TwoTrackerAnalysis&); // Purposefully not implemented.

private:
};


} // namespace


#endif // niftkTwoTrackerAnalysis_h

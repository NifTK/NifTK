/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkVideoTrackerMatching_h
#define niftkVideoTrackerMatching_h

#include "niftkOpenCVExports.h"
#include <cv.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>

namespace mitk
{


class NIFTKOPENCV_EXPORT TrackingMatrices
{
public:
  std::vector<cv::Mat> m_TrackingMatrices;
};
class NIFTKOPENCV_EXPORT TrackingMatrixTimeStamps
{
public:
  std::vector<unsigned long> m_TimeStamps;
  unsigned long GetNearestTimeStamp (unsigned long timestamp , long * delta = NULL );
};

/**
 * \brief A class to match video frames to tracking frames, when reading 
 * recorded tracking data. 
 */
class NIFTKOPENCV_EXPORT VideoTrackerMatching : public itk::Object
{
public: 
  mitkClassMacro ( VideoTrackerMatching, itk::Object);
  itkNewMacro (VideoTrackerMatching);

  /**
   * \brief Initialise the class by passing it a directory name
   */

  void Initialise (std::string directory);

  /**
   * \brief Return the tracking matrix for a given video frame number
   */
  cv::Mat* GetTrackerMatrix ( int TrackerIndex = 0, int * TimingError = NULL ); 
protected:
  VideoTrackerMatching();
  virtual ~VideoTrackerMatching();

  VideoTrackerMatching(const VideoTrackerMatching&); // Purposefully not implemented.
  VideoTrackerMatching& operator=(const VideoTrackerMatching&); // Purposefully not implemented.

private:
  std::vector<unsigned int>             m_FrameNumbers;
  std::vector<TrackingMatrices>         m_TrackingMatrices; 
  std::vector<TrackingMatrixTimeStamps> m_TrackingMatrixTimeStamps; 
  std::string                           m_Directory;
  bool                                  m_Ready;

  std::vector<std::string> FindFrameMaps();
  std::vector<std::string> FindTrackingMatrixDirectories();
  TrackingMatrixTimeStamps FindTrackingTimeStamps(std::string directory);
  
};


} // namespace


#endif // niftkVideoTrackerMatching_h

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkTwoTrackerMatching_h
#define mitkTwoTrackerMatching_h

#include "niftkOpenCVExports.h"
#include <cv.h>
#include <itkObject.h>
#include <itkObjectFactory.h>
#include <mitkCommon.h>
#include "mitkTrackingMatrices.h"
#include "mitkTrackingAndTimeStampsContainer.h"

namespace mitk
{

/**
 * \brief A class to match two sets of time stamped data currently for reading recorded tracking data.
 */
class NIFTKOPENCV_EXPORT TwoTrackerMatching : public itk::Object
{
public: 
  mitkClassMacro ( TwoTrackerMatching, itk::Object);
  itkNewMacro (TwoTrackerMatching);

  /**
   * \brief Initialise the class by passing it two directory names.
   */
  void Initialise (std::string directory1, std::string directory2);

  /**
   * \brief Return the tracking matrix for a given video index.
   */
  cv::Mat GetTrackerMatrix ( unsigned int index, long long * TimingError = NULL, unsigned int TrackerIndex = 0 );

  /**
   * \brief returns state of m_Ready
   */
  bool IsReady () 
  { 
    return m_Ready;
  } 
  
  /**
   * \brief Get the frame count.
   */
  int GetNumberOfFrames () 
  {
    return m_FrameNumbers.size();
  }

  /**
   * \brief If the tracking data is ahead of the video data you can set a video lag in 
   * milliseconds to account for this. If the video is ahead of the tracking set 
   * argument 2 to true
   */
  void SetLagMilliseconds(unsigned long long Lag, bool LagIsNegative =false);

  void FlipMats1 ();
  void FlipMats2 ();

protected:
  TwoTrackerMatching();
  virtual ~TwoTrackerMatching();

  TwoTrackerMatching(const TwoTrackerMatching&); // Purposefully not implemented.
  TwoTrackerMatching& operator=(const TwoTrackerMatching&); // Purposefully not implemented.
  
  std::vector<unsigned int>             m_FrameNumbers;
  TrackingAndTimeStampsContainer        m_TimeStampsContainer1;
  TrackingAndTimeStampsContainer        m_TimeStampsContainer2;
  bool                                  m_Ready;
  std::string                           m_Directory1;
  std::string                           m_Directory2;

  TrackingMatrices                      m_TrackingMatrices11; // The tracking matrices in directory 1
  TrackingMatrices                      m_TrackingMatrices22; // The tracking matrices in directory 2
  TrackingMatrices                      m_TrackingMatrices12; // The tracking matrices in directory 2 corresponding with the timestamps in directory 1
  TrackingMatrices                      m_TrackingMatrices21; // The tracking matrices in directory 1 corresponding with the timestamps in directory 2

private:
  
  void                                  ConvertMatrices(const TrackingAndTimeStampsContainer& container1, TrackingMatrices& container2);
  void                                  LookupMatrices(const TrackingAndTimeStampsContainer& container1,
                                                       const TrackingAndTimeStampsContainer& container2,
                                                       TrackingMatrices& outputContainer);
  bool                                  CheckTimingErrorStats();
  void                                  CreateLookUps();

  unsigned long long                    m_Lag; // The delay between tracker1 and tracker2
  bool                                  m_LagIsNegative; // Controls the direction of lag

  bool                                  m_FlipMat1; // flip matrices in directory1
  bool                                  m_FlipMat2; // flip matrices in directory2

};


} // namespace


#endif // niftkTwoTrackerMatching_h

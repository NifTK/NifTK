/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include "mitkTwoTrackerAnalysis.h"
#include <mitkOpenCVMaths.h>
#include <mitkMathsUtils.h>
#include <mitkOpenCVFileIOUtils.h>
#include <mitkCameraCalibrationFacade.h>

#include <sstream>
#include <fstream>
#include <cstdlib>

namespace mitk 
{
//---------------------------------------------------------------------------
TwoTrackerAnalysis::TwoTrackerAnalysis () 
: m_TimingTolerance(50e6)
{}

//---------------------------------------------------------------------------
TwoTrackerAnalysis::~TwoTrackerAnalysis () 
{}

//---------------------------------------------------------------------------
void TwoTrackerAnalysis::TemporalCalibration(
    int windowLow, int windowHigh, bool visualise, std::string fileout)
{
  if ( !m_Ready )
  {
    MITK_ERROR << "Initialise two tracker matcher before attempting temporal calibration";
    return;
  }

  std::ofstream* fout = new std::ofstream;
  if ( fileout.length() != 0 ) 
  {
    fout->open ( fileout.c_str() );
    if ( !fout )
    {
      MITK_WARN << "Failed to open output file for temporal calibration " << fileout;
    }
  }

  for ( int Lag = windowLow; Lag <= windowHigh ; Lag ++ )
  {
    if ( Lag < 0 ) 
    {
      SetLagMilliseconds ( (unsigned long long) (Lag * -1) , true );
    }
    else 
    {
      SetLagMilliseconds ( (unsigned long long) (Lag ) , false );
    }
   
    if ( m_TimeStampsContainer1.GetSize() > m_TimeStampsContainer2.GetSize() )
    {
      // use 2 
    }
    else
    {
      //use 1
    }
    //then do some kind of correlation between the signals
    //or it may be substantially quicker to not bother with the matcher each time, just 
    //construct two signals and move them back and forwards until there's a match
  }
  //go through the shortest timestamp vector (should be the one with the lowest 
  //mean error), load the matrix, convert to a velocity. Get correlation. How to move??
  //Probably best to use the set lag method

  if ( fileout.length() != 0 ) 
  {
      fout->close();
  }

}
//---------------------------------------------------------------------------
void TwoTrackerAnalysis::HandeyeCalibration(
    bool visualise, std::string fileout, int HowManyMatrices, bool CullOutliers)
{
  if ( !m_Ready )
  {
    MITK_ERROR << "Initialise two tracker matcher before attempting temporal calibration";
    return;
  }

  std::vector<cv::Mat> SortedTracker1;
  std::vector<cv::Mat> SortedTracker2;
  std::vector<int> indexes;
  bool Tracker2ToTracker1 = false;
  //sort distance based on the shortest set. Select up to 80 matrices, evenly spread on distance
  if ( m_TimeStampsContainer1.GetSize() > m_TimeStampsContainer2.GetSize() )
  {
    Tracker2ToTracker1 = false;
    indexes = mitk::SortMatricesByDistance(m_TrackingMatrices22.m_TrackingMatrices);
    for ( unsigned int i = 0; i < indexes.size(); i += indexes.size()/HowManyMatrices )
    {
      long long int timingError;
      
      GetTrackerMatrix(indexes[i],&timingError,1);

      if ( std::abs(timingError) < m_TimingTolerance )
      {
        SortedTracker1.push_back(m_TrackingMatrices22.m_TrackingMatrices[indexes[i]]);
        SortedTracker2.push_back(GetTrackerMatrix(indexes[i],NULL,1).inv());
      }
      else
      {
        MITK_INFO << "Index " << indexes[i] << " Timing error too high, rejecting";
      }

    }
  }
  else
  {
    Tracker2ToTracker1 = true;
    indexes = mitk::SortMatricesByDistance(m_TrackingMatrices11.m_TrackingMatrices);
    for ( unsigned int i = 0; i < indexes.size(); i += indexes.size()/HowManyMatrices )
    {
      long long int timingError;
      
      GetTrackerMatrix(indexes[i],&timingError,0);

      if ( std::abs(timingError) < m_TimingTolerance )
      {
        SortedTracker1.push_back(m_TrackingMatrices11.m_TrackingMatrices[indexes[i]]);
        SortedTracker2.push_back(GetTrackerMatrix(indexes[i],NULL,0).inv());
      }
      else
      {
        MITK_INFO << "Index " << indexes[i] << " Timing error too high, rejecting";
      }
    }
  }
  MITK_INFO << "Starting handeye with " << SortedTracker1.size() << "Matched matrices";

  std::vector <double> residuals;
  cv::Mat w2ToW1 = cvCreateMat(4,4,CV_64FC1);
  cv::Mat t2ToT1 =  Tracker2ToTracker1RotationAndTranslation(SortedTracker1, SortedTracker2,
            residuals, &w2ToW1);
  if ( ! Tracker2ToTracker1 )
  {
    w2ToW1 = w2ToW1.inv();
    t2ToT1 = t2ToT1.inv();
  }
  MITK_INFO << "Handeye finished ";
  MITK_INFO << "Translational Residual " << residuals [1];
  MITK_INFO << "Rotational Residual " << residuals [0];
  
  CheckRigidBody ( w2ToW1, CullOutliers );

  std::string t2ToT1Out = fileout + "_T2ToT1.4x4";
  std::string w2ToW1Out = fileout + "_W2ToW1.4x4";

  if ( ( ! mitk::SaveTrackerMatrix(t2ToT1Out, t2ToT1) ) || 
    ( ! mitk::SaveTrackerMatrix(w2ToW1Out, w2ToW1) ) )
  {
    MITK_ERROR << "Error failed to write out results " << t2ToT1Out << " or " << w2ToW1Out;
  }

}
//---------------------------------------------------------------------------
bool TwoTrackerAnalysis::CheckRigidBody(cv::Mat w2ToW1 , bool CullOutliers)
{
  if ( !m_Ready )
  {
    MITK_ERROR << "Initialise two tracker matcher before attempting temporal calibration";
    return false;
  }

  bool Tracker2ToTracker1 = false;
  std::vector < double > distances;
  if ( m_TimeStampsContainer1.GetSize() > m_TimeStampsContainer2.GetSize() )
  {
    Tracker2ToTracker1 = false;
    for ( unsigned int i = 0; i < m_TrackingMatrices22.m_TrackingMatrices.size() ; i ++  )
    {
      long long int timingError;
      
      GetTrackerMatrix(i,&timingError,1);

      if ( std::abs(timingError) < m_TimingTolerance )
      {
        cv::Mat tracker1 = w2ToW1 * m_TrackingMatrices22.m_TrackingMatrices[i];
        cv::Mat tracker2 = GetTrackerMatrix(i,NULL,1);
        distances.push_back(mitk::DistanceBetweenMatrices(tracker1, tracker2));
      }
      else
      {
        MITK_INFO << "Index " << i << " Timing error too high, rejecting";
      }
    }
  }
  else
  {
    Tracker2ToTracker1 = true;

    for ( unsigned int i = 0; i < m_TrackingMatrices11.m_TrackingMatrices.size() ; i ++  )
    {
      long long int timingError;
      GetTrackerMatrix(i,&timingError,0);

      if ( std::abs(timingError) < m_TimingTolerance )
      {
        cv::Mat tracker1 = m_TrackingMatrices11.m_TrackingMatrices[i];
        cv::Mat tracker2 = w2ToW1 * GetTrackerMatrix(i,NULL,0);
        distances.push_back(mitk::DistanceBetweenMatrices(tracker1, tracker2));
      }
      else
      {
        MITK_INFO << "Index " << i << " Timing error too high, rejecting";
      }
    }
  }
  double meanError = mitk::Mean(distances);
  double stdDev = mitk::StdDev(distances);
  MITK_INFO << "Mean Distance " << meanError ;
  MITK_INFO << "Standard Deviation " << stdDev ;
  if ( CullOutliers )
  {
    if ( ! Tracker2ToTracker1 ) 
    {
      for ( unsigned int i = 0; i < m_TrackingMatrices22.m_TrackingMatrices.size() ; i ++  )
      {
        long long int timingError;
      
        GetTrackerMatrix(i,&timingError,1);

        if ( std::abs(timingError) < m_TimingTolerance )
        {
          cv::Mat tracker1 = w2ToW1 * m_TrackingMatrices22.m_TrackingMatrices[i];
          cv::Mat tracker2 = GetTrackerMatrix(i,NULL,1);
          double distance = mitk::DistanceBetweenMatrices(tracker1, tracker2);
          if ( ( distance > ( meanError + 2 * stdDev )  ) || ( distance < ( meanError - 2 * stdDev) ) ) 
          {
            m_TrackingMatrices21.m_TimingErrors[i] = m_TimingTolerance + 1;
          }
        }
        else
        {
          MITK_INFO << "Index " << i << " Timing error too high, rejecting";
        }
      }
    }
    else
    {
      for ( unsigned int i = 0; i < m_TrackingMatrices11.m_TrackingMatrices.size() ; i ++  )
      {
        long long int timingError;
        GetTrackerMatrix(i,&timingError,0);

        if ( std::abs(timingError) < m_TimingTolerance )
        {
          cv::Mat tracker1 = m_TrackingMatrices11.m_TrackingMatrices[i];
          cv::Mat tracker2 = w2ToW1 * GetTrackerMatrix(i,NULL,0);
          double distance = mitk::DistanceBetweenMatrices(tracker1, tracker2);
          if ( ( distance > ( meanError + 2 * stdDev )  ) || ( distance < ( meanError - 2 * stdDev) ) ) 
          {
            m_TrackingMatrices12.m_TimingErrors[i] = m_TimingTolerance + 1;
          }
        }
        else
        {
          MITK_INFO << "Index " << i << " Timing error too high, rejecting";
        }
      }
    }
  }
  return true;
}

} // namespace

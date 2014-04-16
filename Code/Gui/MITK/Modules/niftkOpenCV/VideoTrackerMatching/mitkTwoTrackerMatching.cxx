/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include "mitkTwoTrackerMatching.h"
#include <mitkCameraCalibrationFacade.h>
#include <mitkOpenCVMaths.h>
#include <mitkOpenCVFileIOUtils.h>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>

#include <sstream>
#include <fstream>
#include <cstdlib>

namespace mitk 
{

//---------------------------------------------------------------------------
TwoTrackerMatching::TwoTrackerMatching () 
: m_Ready(false)
, m_Lag (0)
, m_LagIsNegative(false)
, m_FlipMat1(false)
, m_FlipMat2(false)
{}


//---------------------------------------------------------------------------
TwoTrackerMatching::~TwoTrackerMatching () 
{}


//---------------------------------------------------------------------------
void TwoTrackerMatching::Initialise(std::string directory1, std::string directory2 )
{
  m_Directory1 = directory1;
  m_Directory2 = directory2;
  
  m_TrackingMatrixTimeStamps1 = mitk::FindTrackingTimeStamps(m_Directory1);
  MITK_INFO << "Found " << m_TrackingMatrixTimeStamps1.m_TimeStamps.size() << " time stamped tracking files in " << m_Directory1;
  
  m_TrackingMatrixTimeStamps2 = mitk::FindTrackingTimeStamps(m_Directory2);
  MITK_INFO << "Found " << m_TrackingMatrixTimeStamps2.m_TimeStamps.size() << " time stamped tracking files in " << m_Directory2;

  //now match em up. Do it both ways
  this->CreateLookUps ();

  if ( CheckTimingErrorStats() )
  { 
    MITK_INFO << "TwoTrackerMatching initialised OK";
    m_Ready=true;
    this->LoadOwnMatrices();
  }
  else
  {
    MITK_WARN << "TwoTrackerMatching initialise FAILED";
    m_Ready=false;
  }
  return;
}

//---------------------------------------------------------------------------
void TwoTrackerMatching::CreateLookUps()
{
  cv::Mat trackingMatrix ( 4, 4, CV_64FC1 );
  //do look up from 1 to 2
  m_TrackingMatrices12.m_TimingErrors.clear();
  m_TrackingMatrices12.m_TrackingMatrices.clear();

  for ( unsigned int i = 0 ; i < m_TrackingMatrixTimeStamps1.m_TimeStamps.size() ; i ++ )
  {
    long long timingError;
    unsigned long long TargetTimeStamp;
    if ( m_LagIsNegative )
    {
      TargetTimeStamp = m_TrackingMatrixTimeStamps2.GetNearestTimeStamp(
          m_TrackingMatrixTimeStamps1.m_TimeStamps[i]+m_Lag,&timingError);
    }
    else
    {
      TargetTimeStamp = m_TrackingMatrixTimeStamps2.GetNearestTimeStamp(
          m_TrackingMatrixTimeStamps1.m_TimeStamps[i]-m_Lag,&timingError);
    }
    m_TrackingMatrices12.m_TimingErrors.push_back(timingError);
    std::string MatrixFileName = boost::lexical_cast<std::string>(TargetTimeStamp) + ".txt";
    boost::filesystem::path MatrixFileNameFull (m_Directory2);
    MatrixFileNameFull /= MatrixFileName;
  
    mitk::ReadTrackerMatrix(MatrixFileNameFull.string(), trackingMatrix);

    cv::Mat tmpMatrix ( 4, 4, CV_64FC1 );
    trackingMatrix.copyTo(tmpMatrix);
    
    m_TrackingMatrices12.m_TrackingMatrices.push_back(tmpMatrix);
  }
  //do look up from 2 to 1
  m_TrackingMatrices21.m_TimingErrors.clear();
  m_TrackingMatrices21.m_TrackingMatrices.clear();

  for ( unsigned int i = 0 ; i < m_TrackingMatrixTimeStamps2.m_TimeStamps.size() ; i ++ )
  {
    long long timingError;
    unsigned long long TargetTimeStamp;
    if ( m_LagIsNegative )
    {
      TargetTimeStamp = m_TrackingMatrixTimeStamps1.GetNearestTimeStamp(
          m_TrackingMatrixTimeStamps2.m_TimeStamps[i]-m_Lag,&timingError);
    }
    else
    {
      TargetTimeStamp = m_TrackingMatrixTimeStamps1.GetNearestTimeStamp(
          m_TrackingMatrixTimeStamps2.m_TimeStamps[i]+m_Lag,&timingError);
    }
    m_TrackingMatrices21.m_TimingErrors.push_back(timingError);
    std::string MatrixFileName = boost::lexical_cast<std::string>(TargetTimeStamp) + ".txt";
    boost::filesystem::path MatrixFileNameFull (m_Directory1);
    MatrixFileNameFull /= MatrixFileName;
  
    mitk::ReadTrackerMatrix(MatrixFileNameFull.string(), trackingMatrix);

    cv::Mat tmpMatrix ( 4, 4, CV_64FC1 );
    trackingMatrix.copyTo(tmpMatrix);
    
    m_TrackingMatrices21.m_TrackingMatrices.push_back(tmpMatrix);
  }

}
//---------------------------------------------------------------------------
void TwoTrackerMatching::LoadOwnMatrices()
{
  cv::Mat trackingMatrix ( 4, 4, CV_64FC1 );
  m_TrackingMatrices11.m_TimingErrors.clear();
  m_TrackingMatrices11.m_TrackingMatrices.clear();

  for ( unsigned int i = 0 ; i < m_TrackingMatrixTimeStamps1.m_TimeStamps.size() ; i ++ )
  {
    m_TrackingMatrices11.m_TimingErrors.push_back(0);
    std::string MatrixFileName = boost::lexical_cast<std::string>(m_TrackingMatrixTimeStamps1.m_TimeStamps[i]) + ".txt";
    boost::filesystem::path MatrixFileNameFull (m_Directory1);
    MatrixFileNameFull /= MatrixFileName;
  
    mitk::ReadTrackerMatrix(MatrixFileNameFull.string(), trackingMatrix);

    cv::Mat tmpMatrix ( 4, 4, CV_64FC1 );
    trackingMatrix.copyTo(tmpMatrix);
    
    m_TrackingMatrices11.m_TrackingMatrices.push_back(tmpMatrix);
  }
  m_TrackingMatrices22.m_TimingErrors.clear();
  m_TrackingMatrices22.m_TrackingMatrices.clear();

  for ( unsigned int i = 0 ; i < m_TrackingMatrixTimeStamps2.m_TimeStamps.size() ; i ++ )
  {
    m_TrackingMatrices22.m_TimingErrors.push_back(0);
    std::string MatrixFileName = boost::lexical_cast<std::string>(m_TrackingMatrixTimeStamps2.m_TimeStamps[i]) + ".txt";
    boost::filesystem::path MatrixFileNameFull (m_Directory2);
    MatrixFileNameFull /= MatrixFileName;
  
    mitk::ReadTrackerMatrix(MatrixFileNameFull.string(), trackingMatrix);

    cv::Mat tmpMatrix ( 4, 4, CV_64FC1 );
    trackingMatrix.copyTo(tmpMatrix);
    
    m_TrackingMatrices22.m_TrackingMatrices.push_back(tmpMatrix);
  }
}

//---------------------------------------------------------------------------
void TwoTrackerMatching::SetLagMilliseconds ( unsigned long long Lag, bool VideoLeadsTracking) 
{
  m_Lag = Lag * 1e6;
  m_LagIsNegative = VideoLeadsTracking;

  if ( m_Ready ) 
  {
    MITK_INFO << "Set lag after initialisation reprocessing";

    if ( CheckTimingErrorStats() )
    { 
      this->CreateLookUps();
      MITK_INFO << "TwoTrackerMatching initialised OK";
      m_Ready=true;
    }
    else
    {
      MITK_WARN << "TwoTrackerMatching initialise FAILED";
      m_Ready=false;
    }
  }
  return;
}


//---------------------------------------------------------------------------
bool TwoTrackerMatching::CheckTimingErrorStats()
{
  bool ok = true;
  //check sizes
  if ( m_TrackingMatrices12.m_TrackingMatrices.size() != 
        m_TrackingMatrices12.m_TimingErrors.size() )
  {
    MITK_ERROR << "Wrong number of tracking matrices " << m_TrackingMatrices12.m_TrackingMatrices.size() 
        << " != " <<  m_TrackingMatrices12.m_TimingErrors.size();
      ok = false;
  }


  if ( m_TrackingMatrices21.m_TrackingMatrices.size() != 
        m_TrackingMatrixTimeStamps2.m_TimeStamps.size() )
  {
      MITK_ERROR << "Wrong number of tracking matrix " << ": " << m_TrackingMatrices21.m_TrackingMatrices.size() 
        << " != " <<  m_TrackingMatrixTimeStamps2.m_TimeStamps.size();
      ok = false;
  }
  if ( m_TrackingMatrices21.m_TrackingMatrices.size() != 
        m_TrackingMatrices21.m_TimingErrors.size() )
  {
    MITK_ERROR << "Wrong number of tracking matrices " << m_TrackingMatrices21.m_TrackingMatrices.size() 
        << " != " <<  m_TrackingMatrices21.m_TimingErrors.size();
      ok = false;
  }


  if ( m_TrackingMatrices12.m_TrackingMatrices.size() != 
        m_TrackingMatrixTimeStamps1.m_TimeStamps.size() )
  {
      MITK_ERROR << "Wrong number of tracking matrix " << ": " << m_TrackingMatrices12.m_TrackingMatrices.size() 
        << " != " <<  m_TrackingMatrixTimeStamps2.m_TimeStamps.size();
      ok = false;
  }


  double mean = 0 ; 
  double absmean = 0 ; 
  long long minimum = m_TrackingMatrices12.m_TimingErrors[0];
  long long maximum = m_TrackingMatrices12.m_TimingErrors[0];

  for (unsigned int j = 0 ; j < m_TrackingMatrices12.m_TimingErrors.size() ; j ++ ) 
  {
    mean += static_cast<double>(m_TrackingMatrices12.m_TimingErrors[j]);
    absmean += fabs(static_cast<double>(m_TrackingMatrices12.m_TimingErrors[j]));
    minimum = m_TrackingMatrices12.m_TimingErrors[j] < minimum ? m_TrackingMatrices12.m_TimingErrors[j] : minimum;
    maximum = m_TrackingMatrices12.m_TimingErrors[j] > maximum ? m_TrackingMatrices12.m_TimingErrors[j] : maximum;

  }
  mean /= m_TrackingMatrices12.m_TimingErrors.size();
  absmean /= m_TrackingMatrices12.m_TimingErrors.size();
    
  MITK_INFO << "There are " << m_TrackingMatrices12.m_TimingErrors.size() << " matched frames in for directory 1 to 2";
  MITK_INFO << "Average timing error = " << mean * 1e-6 << "ms";
  MITK_INFO << "Average absolute timing error  = " << absmean * 1e-6 << "ms";
  MITK_INFO << "Maximum timing error = " << maximum * 1e-6 << "ms";
  MITK_INFO << "Minimum timing error = " << minimum * 1e-6 << "ms";

  mean = 0 ; 
  absmean = 0 ; 
  minimum = m_TrackingMatrices21.m_TimingErrors[0];
  maximum = m_TrackingMatrices21.m_TimingErrors[0];

  for (unsigned int j = 0 ; j < m_TrackingMatrices21.m_TimingErrors.size() ; j ++ ) 
  {
    mean += static_cast<double>(m_TrackingMatrices21.m_TimingErrors[j]);
    absmean += fabs(static_cast<double>(m_TrackingMatrices21.m_TimingErrors[j]));
    minimum = m_TrackingMatrices21.m_TimingErrors[j] < minimum ? m_TrackingMatrices21.m_TimingErrors[j] : minimum;
    maximum = m_TrackingMatrices21.m_TimingErrors[j] > maximum ? m_TrackingMatrices21.m_TimingErrors[j] : maximum;
  }
  mean /= m_TrackingMatrices21.m_TimingErrors.size();
  absmean /= m_TrackingMatrices21.m_TimingErrors.size();
    
  MITK_INFO << "There are " << m_TrackingMatrices21.m_TimingErrors.size() << " matched frames in for directory 2 to 1";
  MITK_INFO << "Average timing error = " << mean * 1e-6 << "ms";
  MITK_INFO << "Average absolute timing error  = " << absmean * 1e-6 << "ms";
  MITK_INFO << "Maximum timing error = " << maximum * 1e-6 << "ms";
  MITK_INFO << "Minimum timing error = " << minimum * 1e-6 << "ms";

  return ok;
}
//---------------------------------------------------------------------------
void TwoTrackerMatching::FlipMats1 ( )
{
  if ( !m_Ready ) 
  {
    MITK_WARN << "Attempted to flip matrix 1 when videoTrackerMatching not initialised.";
    return;
  }
  if ( m_FlipMat1 )
  {
    MITK_WARN << "Called flip mat 1 but already done";
    return;
  }
  m_TrackingMatrices11.m_TrackingMatrices = mitk::FlipMatrices(m_TrackingMatrices11.m_TrackingMatrices);
  m_TrackingMatrices21.m_TrackingMatrices = mitk::FlipMatrices(m_TrackingMatrices11.m_TrackingMatrices);
}
//---------------------------------------------------------------------------
void TwoTrackerMatching::FlipMats2 ( )
{
  if ( !m_Ready ) 
  {
    MITK_WARN << "Attempted to flip matrix 2 when videoTrackerMatching not initialised.";
    return;
  }
  if ( m_FlipMat2 )
  {
    MITK_WARN << "Called flip mat 2 but already done";
    return;
  }
  m_TrackingMatrices22.m_TrackingMatrices = mitk::FlipMatrices(m_TrackingMatrices11.m_TrackingMatrices);
  m_TrackingMatrices11.m_TrackingMatrices = mitk::FlipMatrices(m_TrackingMatrices11.m_TrackingMatrices);
}


//---------------------------------------------------------------------------
cv::Mat TwoTrackerMatching::GetTrackerMatrix ( unsigned int FrameNumber , long long * TimingError  ,unsigned int TrackerIndex  )
{
  cv::Mat returnMat = cv::Mat(4,4,CV_64FC1);
  
  if ( !m_Ready ) 
  {
    MITK_WARN << "Attempted to get tracking matrix when videoTrackerMatching not initialised.";
    return returnMat;
  }

  if ( TrackerIndex == 0 )
  {

    if ( FrameNumber >= m_TrackingMatrices12.m_TrackingMatrices.size() )
    {
      MITK_WARN << "Attempted to get tracking matrix with invalid frame index";
      return returnMat;
    }

    returnMat=m_TrackingMatrices12.m_TrackingMatrices[FrameNumber];
    if ( TimingError != NULL ) 
    {
      *TimingError = m_TrackingMatrices12.m_TimingErrors[FrameNumber];
    }
  }
  else
  {
    if ( FrameNumber >= m_TrackingMatrices21.m_TrackingMatrices.size() )
    {
      MITK_WARN << "Attempted to get tracking matrix with invalid frame index";
      return returnMat;
    }

    returnMat=m_TrackingMatrices21.m_TrackingMatrices[FrameNumber];
    if ( TimingError != NULL ) 
    {
      *TimingError = m_TrackingMatrices21.m_TimingErrors[FrameNumber];
    }
  }
 
  return returnMat;
}

} // namespace

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
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include <sstream>
#include <fstream>
#include <cstdlib>

namespace mitk 
{
//---------------------------------------------------------------------------
TwoTrackerAnalysis::TwoTrackerAnalysis () 
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
    
    //then do some kind of correlation between the signals
    //or it may be substantially quicker to not bother with the matcher each time, just 
    //construct two signals and move them back and forwards until there's a match
  }

  if ( fileout.length() != 0 ) 
  {
      fout->close();
  }

}
//---------------------------------------------------------------------------
void TwoTrackerAnalysis::HandeyeCalibration(
    bool visualise, std::string fileout)
{
  MITK_ERROR << "TwoTrackerAnalysis::OptimiseHandeyeCalibration is currently broken, do not use";
  return;
  if ( !m_Ready )
  {
    MITK_ERROR << "Initialise two tracker matcher before attempting temporal calibration";
    return;
  }

  std::ofstream fout;
  if ( fileout.length() != 0 ) 
  {
    fout.open(fileout.c_str());
    if ( !fout )
    {
      MITK_WARN << "Failed to open output file for handeye calibration " << fileout;
    }
  }
}
} // namespace

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <mitkTestingMacros.h>
#include <mitkLogMacros.h>
#include <mitkHandeyeCalibrate.h>
#include <mitkCameraCalibrationFacade.h>


/**
 * Runs ICP registration a known data set and checks the error
 */

int mitkHandeyeCalibrationTest ( int argc, char * argv[] )
{
  std::string inputExtrinsic = argv[1];
  std::string inputTracking = argv[2];
  std::string sort = argv[3];
  std::string result = argv[4];

  mitk::HandeyeCalibrate::Pointer Calibrator = mitk::HandeyeCalibrate::New();

  std::vector<double> Residuals;
  if ( sort == "Distances" )
  {
    Residuals = Calibrator->Calibrate( inputTracking, inputExtrinsic,
        true, false, true, false,result);
  }
  else 
  {
    if ( sort == "Angles" ) 
    {
      Residuals = Calibrator->Calibrate( inputTracking, inputExtrinsic,
        true, false, false, true,result);
    }
    else 
    {
      Residuals = Calibrator->Calibrate( inputTracking, inputExtrinsic,
        true, false, false, false,result);
    }
  }

  int ok = EXIT_SUCCESS;
  double precision = 1e-3;

  for ( unsigned int i = 0 ; i < Residuals.size() ; i ++ )
  {
    if ( fabs(Residuals[i]) > precision )
    {
      ok = EXIT_FAILURE;
      std::cout << "FAIL: " << i << " " << Residuals[i] << " > "  << precision << std::endl;
    }
    else
    {
      std::cout << "PASS: " << i << " " << Residuals[i] << " < "  << precision << std::endl;
    }
  }

  return ok;
}

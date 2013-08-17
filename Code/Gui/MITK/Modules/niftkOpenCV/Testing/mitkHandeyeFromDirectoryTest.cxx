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
#include <mitkHandeyeCalibrateFromDirectory.h>
#include <mitkCameraCalibrationFacade.h>


/**
 * Runs ICP registration a known data set and checks the error
 */

int mitkHandeyeFromDirectoryTest ( int argc, char * argv[] )
{
  
  bool ok = false;
  mitk::HandeyeCalibrateFromDirectory::Pointer Calibrator = mitk::HandeyeCalibrateFromDirectory::New();
  Calibrator->SetDirectory(argv[1]);
  argv++;
  argc--;
  while ( argc > 1 )
  {
    bool argok = false;
    if (( argok == false ) && strcmp ( argv[1], "-NoVideoSupport" ) == 0 ) 
    {
      MITK_WARN << "Test with No video support not implemented.";
      exit(0);
    }
    if ( argok == false )
    {
      MITK_WARN << "Bad parameters.";
      exit (1) ;
    }
  }
                          
  Calibrator->SetTrackerIndex(2);
  Calibrator->SetAbsTrackerTimingError(40e6);
  Calibrator->SetFramesToUse(30);
  Calibrator->SetSortByDistance(true);
  Calibrator->InitialiseVideo();
  return ok;
}

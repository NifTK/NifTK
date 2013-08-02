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
  Calibrator->SetTrackerIndex(2);
  Calibrator->SetFramesToUse(10);
  Calibrator->InitialiseVideo();
  return ok;
}

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

#include <niftkFileHelper.h>
#include <mitkTestingMacros.h>
#include <mitkLogMacros.h>
#include <mitkProjectCameraRays.h>
#include <cmath>

/**
 * \file Test harness for project camera rays
 */
int mitkProjectCameraRaysTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkProjectCameraRaysTest");

  std::string cameraCalibrationFile = argv[1];

  mitk::ProjectCameraRays::Pointer projector = mitk::ProjectCameraRays::New();

  projector->SetIntrinsicFileName(cameraCalibrationFile);


  int returnStatus = projector->Project();
  if ( returnStatus ) 
  {
    projector->WriteOutput ( "/dev/shm/results.txt" );
  }

  if (returnStatus != EXIT_SUCCESS)
  {
    return returnStatus;
  }
  MITK_TEST_END();
}



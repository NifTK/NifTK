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
#include <mitkOpenCVMaths.h>
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
  MITK_TEST_CONDITION_REQUIRED ( returnStatus == true , "Testing that project returned true when no lens to world given." );

  std::vector < std::pair < cv::Point3d , cv::Point3d > > rays = projector->GetRays();

  MITK_TEST_CONDITION_REQUIRED ( mitk::NearlyEqual ( rays[0].first, cv::Point3d ( 0.0, 0.0, 0.0 ) ,1e-6 ) , "Testing first ray origin is zero" );
  MITK_TEST_CONDITION_REQUIRED ( mitk::NearlyEqual ( rays[383].second , cv::Point3d (-241.78379777,
          -139.74497300, 500 ), 1e-6 ), "Testing 383rd ray tip position" << rays[383].second );

  projector = mitk::ProjectCameraRays::New();
  projector->SetIntrinsicFileName(cameraCalibrationFile);
  projector->SetLensToWorldFileName(argv[2]);
  projector->SetProjectAllScreenPoints(false);
  returnStatus = projector->Project();
  MITK_TEST_CONDITION_REQUIRED ( returnStatus == true , "Testing that project returned true when lens to world given" );
  rays = projector->GetRays();
  MITK_TEST_CONDITION_REQUIRED ( mitk::NearlyEqual ( rays[0].first, cv::Point3d ( -129.59307861 , 461.2430114746, -1999.4892578 )
        ,1e-6 ) , "Testing first ray origin is at origin of transform" );
  MITK_TEST_CONDITION_REQUIRED ( mitk::NearlyEqual ( rays[4913].second , cv::Point3d (-4.67841779e+02, 1.35454289e+01, -1.88484736e+03 ), 5e-1 ), "Testing 4913th ray tip position after transformation" <<  rays[4913].second );
  if ( returnStatus )
  {
    projector->WriteOutput ( argv[3] );
  }

  projector->SetLensToWorldFileName("nonsense.nonsense");
  returnStatus = projector->Project();
  MITK_TEST_CONDITION_REQUIRED ( returnStatus == false , "Testing that project returned false when nonsense lens to world given." );

  projector = mitk::ProjectCameraRays::New();
  projector->SetIntrinsicFileName(cameraCalibrationFile);
  projector->SetWidthDivider(21);
  try
  {
    returnStatus = projector->Project();
    MITK_TEST_CONDITION_REQUIRED ( false, "Project should throw an exception when the width is not divisible by width divider");
  }
  catch ( std::exception e )
  {
    MITK_TEST_CONDITION_REQUIRED ( true, "Project threw an exception when the width is not divisible by width divider. " << e.what() );
  }



  MITK_TEST_END();
}



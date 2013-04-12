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

/**
 * Runs ICP registration a known data set and checks the error
 */

int mitkHandeyeCalibrationTest ( int argc, char * argv[] )
{
  std::string inputExtrinsic = argv[1];
  std::string inputTracking = argv[2];

  mitk::HandeyeCalibrate::Pointer Calibrator = mitk::HandeyeCalibrate::New();
  
  std::vector<cv::Mat> ExtMatrices = Calibrator->LoadMatricesFromExtrinsicFile(inputExtrinsic);
  std::vector<cv::Mat> TrackMatrices = Calibrator->LoadMatricesFromDirectory(inputTracking);

  cv::Mat CamToMarker = cvCreateMat(4,4,CV_32FC1);
  Calibrator->Calibrate(ExtMatrices, TrackMatrices, CamToMarker);
  return EXIT_SUCCESS;
}

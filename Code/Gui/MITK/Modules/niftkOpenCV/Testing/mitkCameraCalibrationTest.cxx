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
#include <mitkStereoCameraCalibration.h>
#include <mitkCameraCalibrationFromDirectory.h>
#include <cmath>

/**
 * \class CameraCalibrationTest
 * \brief Test class for Camera Calibration, using data from our Viking 3D Stereo Laparoscope.
 */
class CameraCalibrationTest
{

public:

  //-----------------------------------------------------------------------------
  static void TestStereoReprojectionError(const std::string& inputLeft,
                                          const std::string& inputRight,
                                          const int& cornersX,
                                          const int& cornersY,
                                          const float& squareSize,
                                          const std::string& outputFile,
                                          const float& expectedError
                                          )
  {
    MITK_TEST_OUTPUT(<< "Starting TestStereoReprojectionError...");

    mitk::Point2D scaleFactors;
    scaleFactors[0] = 1;
    scaleFactors[1] = 1;

    mitk::StereoCameraCalibration::Pointer calib = mitk::StereoCameraCalibration::New();
    float actualError = calib->Calibrate(inputLeft, inputRight, cornersX, cornersY, 0, squareSize, scaleFactors, outputFile + ".stereo.txt", false);

    double tolerance = 0.01;
    bool isOK = fabs(actualError - expectedError) < tolerance;
    MITK_TEST_CONDITION_REQUIRED(isOK,".. Testing stereo actualError=" << actualError << " against expectedError=" << expectedError << " with tolerance " << tolerance);

    MITK_TEST_OUTPUT(<< "Finished TestStereoReprojectionError...");
  }

  //-----------------------------------------------------------------------------
  static void TestMonoReprojectionError(const std::string& inputLeft,
                                        const int& cornersX,
                                        const int& cornersY,
                                        const float& squareSize,
                                        const std::string& outputFile,
                                        const float& expectedError
                                        )
  {
    MITK_TEST_OUTPUT(<< "Starting TestMonoReprojectionError...");

    mitk::Point2D scaleFactors;
    scaleFactors[0] = 1;
    scaleFactors[1] = 1;

    mitk::CameraCalibrationFromDirectory::Pointer calib = mitk::CameraCalibrationFromDirectory::New();
    float actualError = calib->Calibrate(inputLeft, cornersX, cornersY, squareSize, scaleFactors, outputFile + ".mono.txt", false);

    double tolerance = 0.01;
    bool isOK = fabs(actualError - expectedError) < tolerance;
    MITK_TEST_CONDITION_REQUIRED(isOK,".. Testing mono actualError=" << actualError << " against expectedError=" << expectedError << " with tolerance " << tolerance);

    MITK_TEST_OUTPUT(<< "Finished TestMonoReprojectionError...");
  }

}; // end class


/**
 * \file Test harness for Mono and Stereo Camera Calibration, provided by OpenCV.
 */
int mitkCameraCalibrationTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkCameraCalibrationTest");

  std::string inputLeft = argv[1];
  std::string inputRight = argv[2];
  std::string outputFile = argv[3];
  int cornersX = atoi(argv[4]);
  int cornersY = atoi(argv[5]);
  float squareSize = atof(argv[6]);
  float monoError = atof(argv[7]);
  float stereoError = atof(argv[8]);

  CameraCalibrationTest::TestMonoReprojectionError(inputLeft, cornersX, cornersY, squareSize, outputFile, monoError);
  CameraCalibrationTest::TestStereoReprojectionError(inputLeft, inputRight, cornersX, cornersY, squareSize, outputFile, stereoError);

  MITK_TEST_END();
}



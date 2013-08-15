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
#include <mitkIOUtil.h>
#include <mitkUltrasoundPinCalibration.h>
#include <mitkVector.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <niftkVTKFunctions.h>
#include <mitkOpenCVMaths.h>

/**
 * \class UltrasoundPinCalibrationTest
 * \brief Test class for ultrasound pin calibration.
 */
class UltrasoundPinCalibrationTest
{

public:

  //-----------------------------------------------------------------------------
  static void DoCalibration(
      std::string& directoryOfMatrices,
      std::string& directoryOfPoints,
      std::string& outputMatrixFileName,
      std::string& fileNameToCompareAgainst
      )
  {
    MITK_TEST_OUTPUT(<< "Starting DoCalibration...");

    mitk::Point3D invariantPoint;
    invariantPoint[0] = 0;
    invariantPoint[1] = 0;
    invariantPoint[2] = 0;

    mitk::Point2D scaleFactors;
    scaleFactors[0] = 0.156; // Assuming image is 256 pixels, and depth = 4cm = 40/256 mm per pixel.
    scaleFactors[1] = 0.156;

    std::vector<double> initialTransformation;
    initialTransformation.push_back(0);
    initialTransformation.push_back(0);
    initialTransformation.push_back(0);
    initialTransformation.push_back(0);
    initialTransformation.push_back(0);
    initialTransformation.push_back(0);

    double residualError = 0;
    vtkSmartPointer<vtkMatrix4x4> calibrationMatrix = vtkMatrix4x4::New();

    // Run Calibration.
    mitk::UltrasoundPinCalibration::Pointer calibration = mitk::UltrasoundPinCalibration::New();
    bool successfullyCalibrated = calibration->CalibrateUsingInvariantPointAndFilesInTwoDirectories(
        directoryOfMatrices,
        directoryOfPoints,
        false,
        true,
        initialTransformation,
        invariantPoint,
        scaleFactors,
        residualError,
        *calibrationMatrix
        );

    MITK_TEST_CONDITION_REQUIRED(successfullyCalibrated == true, "Checking calibration was successful, i.e. it ran, it doesn't mean that it is 'good'.");

    bool successfullySaved = niftk::SaveMatrix4x4ToFile(outputMatrixFileName, *calibrationMatrix);
    MITK_TEST_CONDITION_REQUIRED(successfullySaved == true, "Checking saved file successfully to filename:" << outputMatrixFileName);

    vtkSmartPointer<vtkMatrix4x4> comparisonMatrix = NULL;
    comparisonMatrix = niftk::LoadMatrix4x4FromFile(fileNameToCompareAgainst);

    MITK_TEST_CONDITION_REQUIRED(comparisonMatrix != NULL, "Checking comparison matrix is not null");

    for (int i = 0; i < 4; i++)
    {
      for (int j = 0; j < 4; j++)
      {
        MITK_TEST_CONDITION_REQUIRED(mitk::IsCloseToZero(fabs(comparisonMatrix->GetElement(i, j) - calibrationMatrix->GetElement(i, j)),0.01), "Checking element " << i << ", " << j << " is correct, expecting " << comparisonMatrix->GetElement(i,j) << ", but got " << calibrationMatrix->GetElement(i, j));
      }
    }
    MITK_TEST_OUTPUT(<< "Finished DoCalibration...");
  }
};

/**
 * \file Test harness for Ultrasound Pin Calibration
 */
int mitkUltrasoundPinCalibrationTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkUltrasoundPinCalibrationTest");

  std::string directoryOfMatrices(argv[1]);
  std::string directoryOfPoints(argv[2]);
  std::string outputFileName(argv[3]);
  std::string comparisonFileName(argv[4]);

  UltrasoundPinCalibrationTest::DoCalibration(directoryOfMatrices, directoryOfPoints, outputFileName, comparisonFileName);

  MITK_TEST_END();
}

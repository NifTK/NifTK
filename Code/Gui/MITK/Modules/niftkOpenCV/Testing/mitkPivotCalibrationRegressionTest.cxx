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
#include <mitkPivotCalibration.h>
#include <mitkVector.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <niftkVTKFunctions.h>
#include <mitkOpenCVMaths.h>

/**
 * \class PivotCalibrationRegressionTest
 * \brief Regression test class for pivot calibration, such as might be used for a pointer.
 */
class PivotCalibrationRegressionTest
{

public:

  //-----------------------------------------------------------------------------
  static void DoCalibration(
      std::string& directoryOfMatrices,
      std::string& outputMatrixFileName,
      std::string& fileNameToCompareAgainst
      )
  {
    MITK_TEST_OUTPUT(<< "Starting DoCalibration...");

    double residualError = 0;
    vtkSmartPointer<vtkMatrix4x4> calibrationMatrix = vtkMatrix4x4::New();

    // Run Calibration.
    mitk::PivotCalibration::Pointer calibration = mitk::PivotCalibration::New();
    calibration->CalibrateUsingFilesInDirectories(
        directoryOfMatrices,
        residualError,
        *calibrationMatrix
        );

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
 * \file Test harness for the pivot calibration functionality.
 */
int mitkPivotCalibrationRegressionTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkPivotCalibrationRegressionTest");

  std::string directoryOfMatrices(argv[1]);
  std::string outputFileName(argv[2]);
  std::string comparisonFileName(argv[3]);

  PivotCalibrationRegressionTest::DoCalibration(directoryOfMatrices, outputFileName, comparisonFileName);

  MITK_TEST_END();
}

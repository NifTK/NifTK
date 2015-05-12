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
#include <mitkMathsUtils.h>
#include <mitkOpenCVFileIOUtils.h>

/**
 * \class UltrasoundPinCalibrationRegressionTest
 * \brief Regression test class for ultrasound pin calibration.
 */
class UltrasoundPinCalibrationRegressionTest
{

public:

  //-----------------------------------------------------------------------------
  static void DoCalibration(
      std::string& directoryOfMatrices,
      std::string& directoryOfPoints,
      std::string& outputMatrixFileName,
      std::string& fileNameToCompareAgainst,
      std::string& initialParametersFileName
      )
  {
    MITK_TEST_OUTPUT(<< "Starting DoCalibration...");

    std::ifstream initialParamsStream;
    initialParamsStream.open(initialParametersFileName.c_str());
  
    double in;
    std::vector<double> initialTransformation;
    for ( unsigned int i = 0 ; i < 6 ; i ++ )
    {
      initialParamsStream >> in;
      initialTransformation.push_back(in);
    }

    mitk::Point3D invariantPoint;
    for ( unsigned int i = 0 ; i < 6 ; i ++ )
    {
      initialParamsStream >> in;
      invariantPoint[i] = in;
    }

    mitk::Point2D scaleFactors;
    initialParamsStream >> scaleFactors[0];
    initialParamsStream >> scaleFactors[0];

    double timingLag;
    initialParamsStream >> timingLag;
    
    initialParamsStream.close();
    double residualError = 0;
    vtkSmartPointer<vtkMatrix4x4> calibrationMatrix = vtkSmartPointer<vtkMatrix4x4>::New();

    // Run Calibration.
    mitk::UltrasoundPinCalibration::Pointer calibration = mitk::UltrasoundPinCalibration::New();
    bool successfullyCalibrated = false;
    
    calibration->SetOptimiseImageScaleFactors(true);
    calibration->SetImageScaleFactors(scaleFactors);
    calibration->SetOptimiseInvariantPoint(true);
    calibration->SetInvariantPoint(invariantPoint);
    calibration->SetOptimiseTimingLag(false); //timing lag optimisation isn't quite right yet
    calibration->SetTimingLag(timingLag);
  //  calibration->LoadRigidTransformation(initialGuess);
    calibration->SetVerbose(false);

    mitk::TrackingAndTimeStampsContainer trackingData;
    trackingData.LoadFromDirectory(directoryOfMatrices);
    if (trackingData.GetSize() == 0)
    {
      mitkThrow() << "Failed to tracking data from " << directoryOfMatrices << std::endl;
    }
    calibration->SetTrackingData(&trackingData);
    std::vector< std::pair<unsigned long long, cv::Point3d> > pointData = mitk::LoadTimeStampedPoints(directoryOfPoints);
    if (pointData.size() == 0)
    {
      mitkThrow() << "Failed to load point data from " << directoryOfPoints << std::endl;
    }
    calibration->SetPointData(&pointData);

    residualError = calibration->Calibrate();

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
int mitkUltrasoundPinCalibrationRegressionTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkUltrasoundPinCalibrationRegressionTest");

  std::string directoryOfMatrices(argv[1]);
  std::string directoryOfPoints(argv[2]);
  std::string outputFileName(argv[3]);
  std::string comparisonFileName(argv[4]);
  std::string initialParametersFileName(argv[5]);

  UltrasoundPinCalibrationRegressionTest::DoCalibration(directoryOfMatrices, directoryOfPoints, outputFileName, comparisonFileName, initialParametersFileName);

  MITK_TEST_END();
}

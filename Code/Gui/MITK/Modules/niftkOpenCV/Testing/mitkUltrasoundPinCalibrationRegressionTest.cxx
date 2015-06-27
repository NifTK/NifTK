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
#include <mitkOpenCVMaths.h>
#include <mitkOpenCVFileIOUtils.h>
#include <niftkMathsUtils.h>

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

    if ( ! initialParamsStream )
    {
      mitkThrow() << "Failed to open " << initialParametersFileName;
    }
    double in;
    std::vector<double> initialTransformation;
    for ( unsigned int i = 0 ; i < 6 ; i ++ )
    {
      initialParamsStream >> in;
      initialTransformation.push_back(in);
    }
    mitk::Point3D invariantPoint;
    for ( unsigned int i = 0 ; i < 3 ; i ++ )
    {
      initialParamsStream >> invariantPoint[i];
    }
    mitk::Point2D scaleFactors;
    initialParamsStream >> scaleFactors[0];
    initialParamsStream >> scaleFactors[1];

    double timingLag;
    initialParamsStream >> timingLag;
    
    initialParamsStream.close();
    double residualError = 0;

    // Run Calibration.
    mitk::UltrasoundPinCalibration::Pointer calibration = mitk::UltrasoundPinCalibration::New();
    
    calibration->SetOptimiseImageScaleFactors(true);
    calibration->SetImageScaleFactors(scaleFactors);
    calibration->SetOptimiseInvariantPoint(true);
    calibration->SetInvariantPoint(invariantPoint);
    calibration->SetOptimiseTimingLag(false); //timing lag optimisation isn't quite right yet
    calibration->SetTimingLag(timingLag);
    calibration->SetRigidTransformationParameters(initialTransformation);
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

    cv::Matx44d comparisonMatrix;
    mitk::ReadTrackerMatrix ( fileNameToCompareAgainst , comparisonMatrix);

    cv::Matx44d calibrationMatrix = calibration->GetRigidTransformation();

    for (int i = 0; i < 4; i++)
    {
      for (int j = 0; j < 4; j++)
      {
        MITK_TEST_CONDITION_REQUIRED(niftk::IsCloseToZero(fabs(comparisonMatrix(i, j) - calibrationMatrix(i, j)),0.05), "Checking element " << i << ", " << j << " is correct, expecting " << comparisonMatrix(i,j) << ", but got " << calibrationMatrix(i, j));
      }
    }
    MITK_TEST_CONDITION_REQUIRED(niftk::IsCloseToZero(fabs(0.486643 - residualError),0.001), "Checking residual error is correct, expecting " << 0.486643 << ", but got " << residualError);
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

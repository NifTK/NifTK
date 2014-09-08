/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <limits>
#include <mitkUltrasoundPinCalibration.h>
#include <niftkUltrasoundPinCalibrationCLP.h>
#include <niftkVTKFunctions.h>
#include <mitkVector.h>
#include <mitkExceptionMacro.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if ( matrixDirectory.length() == 0 || pointDirectory.length() == 0)
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    int maxNegativeInt = -32767;

    int numberOfInvariantPoints = 0;
    if (invariantPoint1[0] != maxNegativeInt) numberOfInvariantPoints++;
    if (invariantPoint2[0] != maxNegativeInt) numberOfInvariantPoints++;
    if (invariantPoint3[0] != maxNegativeInt) numberOfInvariantPoints++;

    std::cout << "niftkUltrasoundPinCalibration: matrices       = " << matrixDirectory << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: points         = " << pointDirectory << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: output         = " << outputMatrixFile << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: opt scaling    = " <<  optimiseScaling << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: opt inv points = " <<  optimiseInvariantPoints << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: num inv points = " <<  numberOfInvariantPoints << std::endl;

    // Setup.
    mitk::UltrasoundPinCalibration::Pointer calibration = mitk::UltrasoundPinCalibration::New();
    calibration->SetOptimiseScaling(optimiseScaling);
    calibration->InitialiseMillimetresPerPixel(millimetresPerPixel);
    calibration->SetOptimiseInvariantPoints(optimiseInvariantPoints);
    calibration->SetNumberOfInvariantPoints(numberOfInvariantPoints);
    if (invariantPoint1[0] != maxNegativeInt)
    {
      calibration->InitialiseInvariantPoint(0, invariantPoint1);
    }
    if (invariantPoint2[0] != maxNegativeInt)
    {
      calibration->InitialiseInvariantPoint(1, invariantPoint2);
    }
    if (invariantPoint3[0] != maxNegativeInt)
    {
      calibration->InitialiseInvariantPoint(2, invariantPoint3);
    }

    calibration->InitialiseInitialGuess(initialGuess);

    // Do calibration.
    double residualError = 0;
    vtkSmartPointer<vtkMatrix4x4> transformationMatrix = vtkSmartPointer<vtkMatrix4x4>::New();

    residualError = calibration->CalibrateFromDirectories(
        matrixDirectory,
        pointDirectory,
        *transformationMatrix
        );

    // Save matrix.
    if (!niftk::SaveMatrix4x4ToFile(outputMatrixFile, *transformationMatrix))
    {
      std::ostringstream oss;
      oss << "niftkUltrasoundPinCalibration: Failed to save transformation to file:" << outputMatrixFile << std::endl;
      mitkThrow() << oss.str();
    }

    std::cout << "niftkUltrasoundPinCalibration: residual = " << residualError << std::endl;
    std::cout << "niftkUltrasoundPinCalibration: scaling  = " << calibration->GetMillimetresPerPixel() << std::endl;

    returnStatus = EXIT_SUCCESS;
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception:" << e.what();
    returnStatus = EXIT_FAILURE + 1;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:";
    returnStatus = EXIT_FAILURE + 2;
  }

  return returnStatus;
}

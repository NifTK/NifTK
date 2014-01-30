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
#include <mitkUltrasoundPointerCalibration.h>
#include <niftkUltrasoundPointerCalibrationCLP.h>
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
    std::cout << "niftkUltrasoundPointerCalibration: matrices = " << matrixDirectory << std::endl;
    std::cout << "niftkUltrasoundPointerCalibration: points   = " << pointDirectory << std::endl;
    std::cout << "niftkUltrasoundPointerCalibration: output   = " << outputMatrixFile << std::endl;

    // Setup.
    mitk::UltrasoundPointerCalibration::Pointer calibration = mitk::UltrasoundPointerCalibration::New();
    calibration->InitialisePointerOffset(pointerOffset);
    calibration->SetOptimiseScaling(optimiseScaling);
    calibration->InitialiseMillimetresPerPixel(millimetresPerPixel);
    calibration->InitialisePointerTrackerToProbeTrackerTransform(pointerTrackerToProbeTrackerTransform);
    calibration->InitialiseProbeToProbeTrackerTransform(probeToProbeTrackerTransform);
    calibration->InitialiseInitialGuess(initialGuess);

    // Do calibration.
    double residualError = 0;
    vtkSmartPointer<vtkMatrix4x4> transformationMatrix = vtkMatrix4x4::New();

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

    std::cout << "niftkUltrasoundPointerCalibration: residual = " << residualError << std::endl;
    std::cout << "niftkUltrasoundPointerCalibration: scaling  = " << calibration->GetMillimetresPerPixel() << std::endl;

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
    returnStatus = EXIT_FAILURE + 3;
  }

  return returnStatus;
}

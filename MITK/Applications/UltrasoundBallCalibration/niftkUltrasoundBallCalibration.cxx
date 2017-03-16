/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#include <niftkUltrasoundBallCalibrationCLP.h>
#include <niftkUltrasoundProcessing.h>
#include <mitkPoint.h>
#include <mitkVector.h>
#include <mitkExceptionMacro.h>
#include <niftkMITKMathsUtils.h>
#include <niftkFileIOUtils.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <niftkQuaternion.h>
#include <cv.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if (    imageDirectory.length() == 0
       || matrixDirectory.length() == 0
       || rigidMatrixFile.length() == 0
       || scalingMatrixFile.length() == 0
       )
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    mitk::Point2D scaleFactors;
    niftk::RotationTranslation imageToSensor;

    if (ballSize == 0) // Point calibration
    {
      // Match and load all point and tracking data - must throw exceptions on failure.
      niftk::TrackedPointData trackedPoints = niftk::MatchPointAndTrackingDataFromDirectories(imageDirectory, matrixDirectory);

      // Run calibration - must throw exceptions on failure.
      niftk::DoUltrasoundPointCalibration(trackedPoints,  // input data
                                        scaleFactors,  // output scale factors
                                        imageToSensor  // output trasnformation
                                        );
    }
    else // Ball calibration
    {
      // Load all image and tracking data - must throw exceptions on failure.
      niftk::TrackedImageData trackedImages = niftk::LoadImageAndTrackingDataFromDirectories(imageDirectory, matrixDirectory);

      // Run calibration - must throw exceptions on failure.
      niftk::DoUltrasoundBallCalibration(ballSize,      // command line arg
                                   trackedImages,          // input data
                                   scaleFactors,  // output scale factors
                                   imageToSensor  // output trasnformation
                                   );
    }

    // Convert outputs to matrices (only for consistency - I certainly dont mind if this changes).
    vtkSmartPointer<vtkMatrix4x4> outputMatrix = vtkSmartPointer<vtkMatrix4x4>::New();
    niftk::ConvertRotationAndTranslationToMatrix(imageToSensor.first, imageToSensor.second, *outputMatrix);

    bool savedRigid = niftk::SaveVtkMatrix4x4ToFile(rigidMatrixFile, *outputMatrix);
    if (!savedRigid)
    {
      mitkThrow() << "Failed to save image to sensor transformation to file:" << rigidMatrixFile;
    }

    outputMatrix->Identity();
    outputMatrix->SetElement(0, 0, scaleFactors[0]);
    outputMatrix->SetElement(1, 1, scaleFactors[1]);

    bool savedScaling = niftk::SaveVtkMatrix4x4ToFile(scalingMatrixFile, *outputMatrix);
    if (!savedScaling)
    {
      mitkThrow() << "Failed to save scaling transformation to file:" << scalingMatrixFile;
    }

    returnStatus = EXIT_SUCCESS;
  }
  catch (mitk::Exception& e)
  {
    MITK_ERROR << "Caught mitk::Exception: " << e.GetDescription() << ", from:" << e.GetFile() << "::" << e.GetLine() << std::endl;
    returnStatus = EXIT_FAILURE + 1;
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception: " << e.what() << std::endl;
    returnStatus = EXIT_FAILURE + 2;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:" << std::endl;
    returnStatus = EXIT_FAILURE + 3;
  }

  return returnStatus;
}

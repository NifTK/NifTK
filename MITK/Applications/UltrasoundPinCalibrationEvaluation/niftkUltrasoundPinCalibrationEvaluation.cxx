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
#include <mitkUltrasoundPinCalibrationEvaluation.h>
#include <niftkUltrasoundPinCalibrationEvaluationCLP.h>
#include <niftkVTKFunctions.h>
#include <mitkVector.h>
#include <mitkExceptionMacro.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if ( matrixDirectory.length() == 0
       || pointDirectory.length() == 0
       || calibrationMatrix.length() == 0
       || cameraToWorldMatrix.length() == 0
       )
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    mitk::Point3D invPoint;
    invPoint[0] = 0;
    invPoint[1] = 0;
    invPoint[2] = 0;
    if (invariantPoint.size() == 3)
    {
      invPoint[0] = invariantPoint[0];
      invPoint[1] = invariantPoint[1];
      invPoint[2] = invariantPoint[2];
    }

    mitk::Point2D mmPerPix;
    mmPerPix[0] = 1;
    mmPerPix[1] = 1;
    if (millimetresPerPixel.size() == 2)
    {
      mmPerPix[0] = millimetresPerPixel[0];
      mmPerPix[1] = millimetresPerPixel[1];
    }

    mitk::UltrasoundPinCalibrationEvaluation::Pointer evaluator = mitk::UltrasoundPinCalibrationEvaluation::New();
    evaluator->Evaluate(
          matrixDirectory,
          pointDirectory,
          invPoint,
          mmPerPix,
          calibrationMatrix,
          cameraToWorldMatrix
          );
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

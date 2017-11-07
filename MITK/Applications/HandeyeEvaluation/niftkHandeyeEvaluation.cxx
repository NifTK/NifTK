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
#include <mitkExceptionMacro.h>
#include <niftkHandeyeEvaluationCLP.h>
#include <niftkHandEyeEvaluationInterface.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  try
  {
    double rms = std::numeric_limits<double>::max();

    int lag = niftk::EvaluateHandeyeFromPoints(trackingInputDirectory,
                                               pointsInputDirectory,
                                               modelFile,
                                               intrinsicsFile,
                                               handeyeFile,
                                               registrationFile,
                                               rms
                                              );

    std::cout << "niftkHandeyeEvaluation: Lag:" << lag << ", rms:" << rms << std::endl;

    returnStatus = EXIT_SUCCESS;
  }
  catch (mitk::Exception& e)
  {
    MITK_ERROR << "Caught mitk::Exception: " << e.GetDescription() << ", from:" << e.GetFile() << "::" << e.GetLine() << std::endl;
    returnStatus = EXIT_FAILURE + 100;
  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception: " << e.what() << std::endl;
    returnStatus = EXIT_FAILURE + 101;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:" << std::endl;
    returnStatus = EXIT_FAILURE + 102;
  }

  return returnStatus;
}

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
#include <niftkUltrasoundTransformAndImageMergerCLP.h>
#include <mitkVector.h>
#include <QApplication>
#include <mitkUltrasoundTransformAndImageMerger.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if (inputMatrixDirectory.length() == 0
      || inputImageDirectory.length() == 0)
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {

    mitk::UltrasoundTransformAndImageMerger::Pointer merger = mitk::UltrasoundTransformAndImageMerger::New();
    merger->Merge(inputMatrixDirectory, inputImageDirectory, outputImageFile);

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

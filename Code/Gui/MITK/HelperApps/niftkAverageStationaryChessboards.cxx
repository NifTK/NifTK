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
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <niftkAverageStationaryChessboardsCLP.h>
#include <niftkFileHelper.h>
#include <mitkOpenCVFileIOUtils.h>
#include <mitkFileIOUtils.h>
#include <mitkExceptionMacro.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if (    leftCameraInputDirectory.length() == 0
       || rightCameraInputDirectory.length() == 0
       || intrinsicLeft.length() == 0
       || intrinsicRight.length() == 0
       || rightToLeftExtrinsics.length() == 0
       || outputPoints.length() == 0
       )
  {
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    std::vector<std::string> leftFiles = niftk::GetFilesInDirectory(leftCameraInputDirectory);
    if (leftFiles.size() == 0)
    {
      std::ostringstream errorMessage;
      errorMessage << "No files in directory:" << leftCameraInputDirectory << std::endl;
      mitkThrow() << errorMessage.str();
    }

    std::vector<std::string> rightFiles = niftk::GetFilesInDirectory(rightCameraInputDirectory);
    if (rightFiles.size() == 0)
    {
      std::ostringstream errorMessage;
      errorMessage << "No files in directory:" << rightCameraInputDirectory << std::endl;
      mitkThrow() << errorMessage.str();
    }

    if (leftFiles.size() != rightFiles.size())
    {
      std::ostringstream errorMessage;
      errorMessage << "Different number of files in left:" << leftCameraInputDirectory << ", and right:" <<  rightCameraInputDirectory << std::endl;
      mitkThrow() << errorMessage.str();
    }

    std::sort(leftFiles.begin(), leftFiles.end());
    std::sort(rightFiles.begin(), rightFiles.end());

    // Done
    returnStatus = EXIT_SUCCESS;
  }
  catch (std::exception& e)
  {
    std::cerr << "Caught std::exception:" << e.what();
    returnStatus = -1;
  }
  catch (...)
  {
    std::cerr << "Caught unknown exception:";
    returnStatus = -2;
  }
  return returnStatus;
}

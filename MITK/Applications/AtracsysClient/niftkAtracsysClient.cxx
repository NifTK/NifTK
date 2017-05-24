/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkAtracsysClientCLP.h>
#include <niftkAtracsysTracker.h>
#include <niftkFileIOUtils.h>
#include <niftkFileHelper.h>
#include <mitkStandaloneDataStorage.h>
#include <mitkException.h>

int main(int argc, char* argv[])
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  try
  {
    if (toolStorage.empty() || outputDir.empty())
    {
      commandLine.getOutput()->usage(commandLine);
      return returnStatus + 1;
    }

    mitk::StandaloneDataStorage::Pointer dataStorage = mitk::StandaloneDataStorage::New();
    niftk::AtracsysTracker tracker(dataStorage, toolStorage);

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

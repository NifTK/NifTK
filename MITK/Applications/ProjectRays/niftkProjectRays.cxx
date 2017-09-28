/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <niftkProjectRaysCLP.h>
#include <mitkProjectCameraRays.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if ( calibration.length() == 0 )
  {
    std::cout << calibration.length() << std::endl;
    commandLine.getOutput()->usage(commandLine);
    return returnStatus;
  }

  try
  {
    mitk::ProjectCameraRays::Pointer projector = mitk::ProjectCameraRays::New();

    projector->SetIntrinsicFileName(calibration);
    if ( lensToWorld.length() != 0 )
    {
      projector->SetLensToWorldFileName(lensToWorld);
    }

    returnStatus = projector->Project();

    if ( returnStatus )
    {
      projector->WriteOutput ( outFile );
    }

  }
  catch (std::exception& e)
  {
    MITK_ERROR << "Caught std::exception:" << e.what();
    returnStatus = -1;
  }
  catch (...)
  {
    MITK_ERROR << "Caught unknown exception:";
    returnStatus = -2;
  }

  return returnStatus;
}

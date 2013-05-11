/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <niftkRegisterProbeModelToStereoPairCLP.h>
#include <mitkRegisterProbeModelToStereoPair.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  bool successful = false;

  if (   inputModel.length() == 0
      || inputLeft.length() == 0
      || inputRight.length() == 0
      || outputLeft.length() == 0
      || outputRight.length() == 0
      || intrinsicLeft.length() == 0
      || distortionLeft.length() == 0
      || rotationLeft.length() == 0
      || translationLeft.length() == 0
      || intrinsicRight.length() == 0
      || distortionRight.length() == 0
      || rightToLeftRotation.length() == 0
      || rightToLeftTranslation.length() == 0
      )
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  mitk::RegisterProbeModelToStereoPair::Pointer registrationObject = mitk::RegisterProbeModelToStereoPair::New();
  successful = registrationObject->DoRegistration(
      inputModel,
      inputLeft,
      inputRight,
      outputLeft,
      outputRight,
      intrinsicLeft,
      distortionLeft,
      rotationLeft,
      translationLeft,
      intrinsicRight,
      distortionRight,
      rightToLeftRotation,
      rightToLeftTranslation,
      rx, ry, rz, tx, ty, tz
      );

  if (successful)
  {
    return EXIT_SUCCESS;
  }
  else
  {
    return EXIT_FAILURE;
  }
}

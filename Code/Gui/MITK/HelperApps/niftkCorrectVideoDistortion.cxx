/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <niftkCorrectVideoDistortionCLP.h>
#include <mitkCorrectVideoFileDistortion.h>
#include <mitkCorrectImageDistortion.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  bool successful = false;

  if ( input.length() == 0
      || intrinsicLeft.length() == 0
      || distortionLeft.length() == 0
      || output.length() == 0
      )
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  if (input == output)
  {
    std::cerr << "Output filename is the same as the input ...  I'm giving up." << std::endl;
    return EXIT_FAILURE;
  }

  bool isVideo = false;
  if (input.find_last_of(".") != std::string::npos)
  {
    std::string extension = input.substr(input.find_last_of(".")+1);
    if (extension == "avi")
    {
      isVideo = true;
    }
  }

  if (isVideo)
  {
    // Only supporting stereo video for now.
    if (   intrinsicRight.length() == 0
        || distortionRight.length() == 0
        )
    {
      std::cerr << "For processing stereo video, the right channel must be specified" << std::endl;
      return EXIT_FAILURE;
    }

    mitk::CorrectVideoFileDistortion::Pointer correction = mitk::CorrectVideoFileDistortion::New();
    successful = correction->Correct(
        input,
        intrinsicLeft,
        distortionLeft,
        intrinsicRight,
        distortionRight,
        output,
        writeInterleaved
        );
  }
  else
  {
    mitk::CorrectImageDistortion::Pointer correction = mitk::CorrectImageDistortion::New();
    successful = correction->Correct(input, intrinsicLeft, distortionLeft, output);
  }

  if (successful)
  {
    return EXIT_SUCCESS;
  }
  else
  {
    return EXIT_FAILURE;
  }
}

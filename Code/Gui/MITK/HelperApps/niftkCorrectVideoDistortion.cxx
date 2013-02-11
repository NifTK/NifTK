/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include "niftkCorrectVideoDistortionCLP.h"
#include "mitkCorrectVideoFileDistortion.h"

int main(int argc, char** argv)
{
  PARSE_ARGS;
  bool successful = false;

  if ( inputVideo.length() == 0
      || inputIntrinsicParams.length() == 0
      || inputDistortionParams.length() == 0
      || outputVideo.length() == 0
      )
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  if (   inputVideo.size() == 0
      || inputIntrinsicParams.size() == 0
      || inputDistortionParams.size() == 0
      || outputVideo.size() == 0
      )
  {
    throw std::logic_error("Empty filename supplied");
  }

  if (inputVideo == outputVideo)
  {
    throw std::logic_error("Output filename is the same as the input ...  I'm giving up.");
  }

  mitk::CorrectVideoFileDistortion::Pointer correction = mitk::CorrectVideoFileDistortion::New();

  bool isVideo = false;
  if (inputVideo.find_last_of(".") != std::string::npos)
  {
    std::string extension = inputVideo.substr(inputVideo.find_last_of(".")+1);
    if (extension == "avi")
    {
      isVideo = true;
    }
  }

  successful = correction->Correct(inputVideo, inputIntrinsicParams, inputDistortionParams, outputVideo, isVideo);

  if (successful)
  {
    return EXIT_SUCCESS;
  }
  else
  {
    return EXIT_FAILURE;
  }
}

/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include "niftkTrackLapUSProbeCLP.h"
#include "mitkTrackLapUS.h"

int main(int argc, char** argv)
{
  PARSE_ARGS;
  bool successful = false;

  if ( input.length() == 0
      || intrinsicLeft.length() == 0
      || distortionLeft.length() == 0
      || output.length() == 0
      || intrinsicRight.length() == 0
      || distortionRight.length() == 0
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

  mitk::TrackLapUS::Pointer tracker = mitk::TrackLapUS::New();
  successful = tracker->Track(
      input,
      intrinsicLeft,
      distortionLeft,
      intrinsicRight,
      distortionRight,
      output,
      writeInterleaved
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

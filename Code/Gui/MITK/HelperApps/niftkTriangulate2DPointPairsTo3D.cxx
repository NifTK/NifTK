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
#include <mitkTriangulate2DPointPairsTo3D.h>
#include <niftkTriangulate2DPointPairsTo3DCLP.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;

  if ( inputPointPairs.length() == 0
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
    mitk::Triangulate2DPointPairsTo3D::Pointer triangulator = mitk::Triangulate2DPointPairsTo3D::New();
    triangulator->Triangulate(
                  inputPointPairs,
                  intrinsicLeft,
                  intrinsicRight,
                  rightToLeftExtrinsics,
                  outputPoints
                  );

    returnStatus = EXIT_SUCCESS;
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

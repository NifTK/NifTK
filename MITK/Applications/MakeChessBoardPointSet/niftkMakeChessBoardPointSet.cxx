/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <mitkPointSet.h>
#include <mitkIOUtil.h>

#include <mitkPointUtils.h>
#include <niftkConversionUtils.h>
#include <niftkMakeChessBoardPointSetCLP.h>

/**
 * \brief Generates a 2D image with a calibration pattern.
 */
int main(int argc, char** argv)
{
  // To parse command line args.
  PARSE_ARGS;

  if ( outputPointSet.length() == 0
      )
  {
    commandLine.getOutput()->usage(commandLine);
    return EXIT_FAILURE;
  }

  mitk::Point3D dx;
  mitk::Point3D dy;

  for (unsigned int i = 0; i < 3; i++)
  {
    dx[i] = right[i] - origin[i];
    dy[i] = down[i] - origin[i];
  }
  mitk::Normalise(dx);
  mitk::Normalise(dy);

  mitk::Point3D point;
  mitk::PointSet::Pointer pointSet = mitk::PointSet::New();
  int counter = 0;

  for (unsigned int r = 0; r < internalCorners[1]; r++)
  {
    for (unsigned int c = 0; c < internalCorners[0]; c++)
    {
      point[0] = origin[0] + dx[0]*c*squareSize + dy[0]*r*squareSize;
      point[1] = origin[1] + dx[1]*c*squareSize + dy[1]*r*squareSize;
      point[2] = origin[2] + dx[2]*c*squareSize + dy[2]*r*squareSize;

      pointSet->InsertPoint(counter, point);
      counter++;
    }
  }

  mitk::IOUtil::Save(pointSet, outputPointSet);

  return EXIT_SUCCESS;
}

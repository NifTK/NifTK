/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <mitkMakeGeometry.h>
#include <mitkIOUtil.h>
#include <niftkIGIMakeGeometryCLP.h>

int main(int argc, char** argv)
{
  PARSE_ARGS;
  int returnStatus = EXIT_FAILURE;
  mitk::Surface::Pointer surface = mitk::Surface::New();
  if ( geometry == "backwall" )
  {
    surface = MakeAWall(0);
  }
  if ( geometry == "frontwall" )
  {
    surface = MakeAWall(2);
  }
  if ( geometry == "leftwall" )
  {
    surface = MakeAWall(1);
  }
  if ( geometry == "rightwall" )
  {
    surface = MakeAWall(3);
  }
  if ( geometry == "ceiling" )
  {
    surface = MakeAWall(4);
  }
  if ( geometry == "floor" )
  {
    surface = MakeAWall(5);
  }
  if ( geometry == "laparoscope" )
  {
    surface = MakeLaparoscope (rigidBodyFile, handeye);
  }

  mitk::IOUtil::SaveSurface (surface,output);


}

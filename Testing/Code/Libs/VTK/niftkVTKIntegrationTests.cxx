/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include <iostream>
#include <itkTestMain.h>

void RegisterTests()
{
  REGISTER_TEST(VTKIterativeClosestPointTest);
  REGISTER_TEST(VTKIterativeClosestPointRepeatTest);
  REGISTER_TEST(VTK3PointReaderTest);
  REGISTER_TEST(VTK4PointReaderTest);
  REGISTER_TEST(VTKDistanceToSurfaceTest);
  REGISTER_TEST(VTKDistanceToSurfaceTestSinglePoint);
}


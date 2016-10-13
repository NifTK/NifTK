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
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <niftkMathsUtils.h>

int testMahalanobisDistance()
{
  std::vector<double> v1;
  std::vector<double> v2;
  std::vector<double> cov;

  niftk::CheckDoublesEquals(0.0, niftk::MahalanobisDistance(v1, v2, cov), 0.0001);
  return EXIT_SUCCESS;
}

/**
 * Basic test harness for MathsUtils.h
 */
int niftkMathsUtilsTest(int argc, char * argv[])
{
  if (argc < 2)
    {
      std::cerr << "Usage   :niftkMathsUtilsTest testNumber" << std::endl;
      return 1;
    }

  int testNumber = atoi(argv[1]);

  if (testNumber == 1)
    {
      return testMahalanobisDistance();
    }
  else
    {
      return EXIT_FAILURE;
    }
}


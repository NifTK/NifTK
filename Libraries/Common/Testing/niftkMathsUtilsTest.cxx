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
  v1.push_back ( 0.0 );
  try
  {
    niftk::CheckDoublesEquals(0.0, niftk::MahalanobisDistance(v1, v2, cov), 0.0001);
    return EXIT_FAILURE;
  }
  catch (...)
  {
  }

  v2.push_back (1.0);
  cov.push_back (0.0);

  try
  {
    niftk::CheckDoublesEquals(0.0, niftk::MahalanobisDistance(v1, v2, cov), 0.0001);
    return EXIT_FAILURE;
  }
  catch (...)
  {
  }

  cov[0] = 0.5;
  try
  {
    niftk::CheckDoublesEquals(sqrt(2.0), niftk::MahalanobisDistance(v1, v2, cov), 0.0001);
  }
  catch (...)
  {
    return EXIT_FAILURE;
  }

  v1.push_back(-3.0);
  v2.push_back(-1.0);
  cov.push_back(2.0);

  try
  {
    niftk::CheckDoublesEquals(2.0, niftk::MahalanobisDistance(v1, v2, cov), 0.0001);
  }
  catch (...)
  {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int testCheckDoublesEquals()
{
  try
  {
    niftk::CheckDoublesEquals(0.0, 0.0001, 0.0001);
  }
  catch (...)
  {
    return EXIT_FAILURE;
  }

  try
  {
    niftk::CheckDoublesEquals(-1.0, -1.0001, 0.0001);
  }
  catch (...)
  {
    return EXIT_FAILURE;
  }

  try
  {
    niftk::CheckDoublesEquals(0.0, 0.00011, 0.0001);
    return EXIT_FAILURE;
  }
  catch (...)
  {
  }

  try
  {
    niftk::CheckDoublesEquals(-1.0, -1.00011, 0.0001);
    return EXIT_FAILURE;
  }
  catch (...)
  {
  }
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
  else if (testNumber == 2)
    {
      return testCheckDoublesEquals();
    }
  else
    {
      return EXIT_FAILURE;
    }
}


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
#include "ConversionUtils.h"
#include <math.h>
#include <iostream>
#include "stdlib.h"

void CheckDoublesEquals(double expected, double actual, double tol)
{
  if (fabs(expected - actual) > tol)
    {
      std::cerr << "Failed:Expected=" << expected << ", actual=" << actual << ", tolerance=" << tol << std::endl;
      throw std::exception();
    }
}

int testCalculateVarianceFromFWHM()
{
  CheckDoublesEquals(2.88539008177793, niftk::CalculateVarianceFromFWHM(4.0), 0.0001);
  CheckDoublesEquals(0, niftk::CalculateVarianceFromFWHM(0), 0.0001);
  return EXIT_SUCCESS;
}

int testCalculateStdDevFromFWHM()
{
  CheckDoublesEquals(1.69864360057604, niftk::CalculateStdDevFromFWHM(4.0) , 0.0001);
  CheckDoublesEquals(0, niftk::CalculateStdDevFromFWHM(0), 0.0001);
  return EXIT_SUCCESS;
}

int testConvertFirstVoxelCoordinateToMiddleOfImageCoordinate()
{
  CheckDoublesEquals(3, niftk::ConvertFirstVoxelCoordinateToMiddleOfImageCoordinate(0,9,0.75), 0.001);  
  CheckDoublesEquals(2.5, niftk::ConvertFirstVoxelCoordinateToMiddleOfImageCoordinate(-0.5,9,0.75), 0.001);
  CheckDoublesEquals(3.5, niftk::ConvertFirstVoxelCoordinateToMiddleOfImageCoordinate(0.5,9,0.75), 0.001);
  CheckDoublesEquals(2.625, niftk::ConvertFirstVoxelCoordinateToMiddleOfImageCoordinate(0,8,0.75), 0.001);  
  CheckDoublesEquals(2.125, niftk::ConvertFirstVoxelCoordinateToMiddleOfImageCoordinate(-0.5,8,0.75), 0.001);
  CheckDoublesEquals(3.125, niftk::ConvertFirstVoxelCoordinateToMiddleOfImageCoordinate(0.5,8,0.75), 0.001);
  CheckDoublesEquals(0.5, niftk::ConvertFirstVoxelCoordinateToMiddleOfImageCoordinate(0.5,1,0.33), 0.001);
  CheckDoublesEquals(0.665, niftk::ConvertFirstVoxelCoordinateToMiddleOfImageCoordinate(0.5,2,0.33), 0.001);
  return EXIT_SUCCESS;
}

int testConvertMiddleOfImageCoordinateToFirstVoxelCoordinate()
{
  CheckDoublesEquals(-3, niftk::ConvertMiddleOfImageCoordinateToFirstVoxelCoordinate(0,9,0.75), 0.001);  
  CheckDoublesEquals(-3.5, niftk::ConvertMiddleOfImageCoordinateToFirstVoxelCoordinate(-0.5,9,0.75), 0.001);
  CheckDoublesEquals(-2.5, niftk::ConvertMiddleOfImageCoordinateToFirstVoxelCoordinate(0.5,9,0.75), 0.001);
  CheckDoublesEquals(-2.625, niftk::ConvertMiddleOfImageCoordinateToFirstVoxelCoordinate(0,8,0.75), 0.001);  
  CheckDoublesEquals(-3.125, niftk::ConvertMiddleOfImageCoordinateToFirstVoxelCoordinate(-0.5,8,0.75), 0.001);
  CheckDoublesEquals(-2.125, niftk::ConvertMiddleOfImageCoordinateToFirstVoxelCoordinate(0.5,8,0.75), 0.001);  
  CheckDoublesEquals(0.5, niftk::ConvertMiddleOfImageCoordinateToFirstVoxelCoordinate(0.5,1,0.33), 0.001);
  CheckDoublesEquals(0.335, niftk::ConvertMiddleOfImageCoordinateToFirstVoxelCoordinate(0.5,2,0.33), 0.001);
  return EXIT_SUCCESS;
}

int testRoundingToNDecimalPlaces()
{
  CheckDoublesEquals(12.35, niftk::Round(12.345,2), 0.001);
  CheckDoublesEquals(12.34, niftk::Round(12.344,2), 0.001);
  return EXIT_SUCCESS;
}

int testGetLastNCharacters()
{
  std::string a("abcdefghijklmno");
  if (niftk::GetLastNCharacters(a, 5) != "klmno") {
    std::cerr << "Expected klmno but got:" << niftk::GetLastNCharacters(a, 5) << std::endl;
    return EXIT_FAILURE;
  }
  if (niftk::GetLastNCharacters(a, 15) != "abcdefghijklmno") {
    std::cerr << "Expected abcdefghijklmno but got:" << niftk::GetLastNCharacters(a, 15) << std::endl;
    return EXIT_FAILURE;
  }
  if (niftk::GetLastNCharacters(a, 20) != "abcdefghijklmno") {
    std::cerr << "Expected abcdefghijklmno but got:" << niftk::GetLastNCharacters(a, 20) << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

/**
 * Basic test harness for ConversionUtils.h
 */
int ConversionUtilsTest(int argc, char * argv[])
{
  if (argc < 2)
    {
      std::cerr << "Usage   :ConversionUtilsTest testNumber" << std::endl;
      return 1;
    }
  
  int testNumber = atoi(argv[1]);
  
  if (testNumber == 1)
    {
      return testCalculateVarianceFromFWHM();
    }
  else if (testNumber == 2)
    {
      return testCalculateStdDevFromFWHM();
    }
  else if (testNumber == 3)
    {
      return testConvertFirstVoxelCoordinateToMiddleOfImageCoordinate(); 
    }
  else if (testNumber == 4)
    {
      return testConvertMiddleOfImageCoordinateToFirstVoxelCoordinate();  
    }
  else if (testNumber == 5)
    {
      return testRoundingToNDecimalPlaces();
    }
  else if (testNumber == 6)
    {
      return testGetLastNCharacters();
    }
  else
    {
      return EXIT_FAILURE;
    }
}


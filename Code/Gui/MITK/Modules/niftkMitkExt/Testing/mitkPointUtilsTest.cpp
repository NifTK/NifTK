/*=============================================================================

 NifTK: An image processing toolkit jointly developed by the
             Dementia Research Centre, and the Centre For Medical Image Computing
             at University College London.

 See:        http://dementia.ion.ucl.ac.uk/
             http://cmic.cs.ucl.ac.uk/
             http://www.ucl.ac.uk/

 Last Changed      : $Date$
 Revision          : $Revision$
 Last modified by  : $Author$

 Original author   : m.clarkson@ucl.ac.uk

 Copyright (c) UCL : See LICENSE.txt in the top level directory for details.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notices for more information.

 ============================================================================*/
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
#include <math.h>
#include <iostream>
#include <cstdlib>
#include <mitkTestingMacros.h>
#include <mitkVector.h>
#include "mitkPointUtils.h"

/**
 * \brief Test class for mitkPointUtils
 */
class mitkPointUtilsTestClass
{

public:

  //-----------------------------------------------------------------------------
  static void TestCalculateStepSize()
  {
    MITK_TEST_OUTPUT(<< "Starting TestCalculateStepSize...");

    double spacing[3];
    double result;

    spacing[0] = 0;
    spacing[1] = 0;
    spacing[2] = 0;
    result = mitk::CalculateStepSize(spacing);
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(result, 0),".. Testing result==0");

    spacing[0] = 1;
    spacing[1] = 1;
    spacing[2] = 1;
    result = mitk::CalculateStepSize(spacing);
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(result, (double)1/(double)3),".. Testing result==1/3");

    spacing[0] = 0.5;
    spacing[1] = 1;
    spacing[2] = 1;
    result = mitk::CalculateStepSize(spacing);
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(result, (double)1/(double)6),".. Testing result==1/6");

    spacing[0] = 1;
    spacing[1] = 1;
    spacing[2] = (double)1/(double)3;
    result = mitk::CalculateStepSize(spacing);
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(result, (double)1/(double)9),".. Testing result==1/9");

    MITK_TEST_OUTPUT(<< "Finished TestCalculateStepSize...");
  }

  //-----------------------------------------------------------------------------
  static void TestAreDifferent()
  {
    MITK_TEST_OUTPUT(<< "Starting TestAreDifferent...");

    mitk::Point3D a;
    mitk::Point3D b;
    bool result;

    // Test for each axis
    for (int i = 0; i < 3; i++)
    {
      a[0] = 0; a[1] = 0; a[2] = 0;
      b[0] = 0; b[1] = 0; b[2] = 0;

      result = mitk::AreDifferent(a,b);
      MITK_TEST_CONDITION_REQUIRED(mitk::Equal(result, false),".. Testing result==false");

      a[i] = 0.009;
      result = mitk::AreDifferent(a,b);
      MITK_TEST_CONDITION_REQUIRED(mitk::Equal(result, false),".. Testing result with tolerance 0.01==false");

      a[i] = 0.011;
      result = mitk::AreDifferent(a,b);
      MITK_TEST_CONDITION_REQUIRED(mitk::Equal(result, true),".. Testing result with tolerance 0.01==true");
    }

    MITK_TEST_OUTPUT(<< "Finished TestAreDifferent...");
  }

  //-----------------------------------------------------------------------------
  static void TestGetSquaredDistanceBetweenPoints()
  {
    MITK_TEST_OUTPUT(<< "Starting TestGetSquaredDistanceBetweenPoints...");

    mitk::Point3D a;
    mitk::Point3D b;

    for (unsigned int i = 0; i < 3; i++)
    {
      for (int j = -1; j<=1; j++)
      {
        a[0] = 0;    a[1] = 0;   a[2] = 0;
        b[0] = i*j;  b[1] = i*j; b[2] = i*j;

        double distance =  (a[0]-b[0])*(a[0]-b[0])
                         + (a[1]-b[1])*(a[1]-b[1])
                         + (a[2]-b[2])*(a[2]-b[2]);
        double result = mitk::GetSquaredDistanceBetweenPoints(a,b);
        MITK_TEST_CONDITION_REQUIRED(mitk::Equal(result, distance),".. Testing GetSquaredDistanceBetweenPoints");
      }
    }
    MITK_TEST_OUTPUT(<< "Finished TestGetSquaredDistanceBetweenPoints...");
  }

  //-----------------------------------------------------------------------------
  static void TestGetDifference()
  {
    MITK_TEST_OUTPUT(<< "Starting TestGetDifference...");

    mitk::Point3D a;
    mitk::Point3D b;
    mitk::Point3D c;

    a[0] = 1; a[1] = 2; a[2] = 3;
    b[0] = 0; b[1] = -2; b[2] = 1000;

    mitk::GetDifference(a, b, c);

    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(c[0], 1),".. Testing TestGetDifference[0]=1");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(c[1], 4),".. Testing TestGetDifference[1]=4");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(c[2], -997),".. Testing TestGetDifference[2]=-997");

    MITK_TEST_OUTPUT(<< "Finished TestGetDifference...");
  }


  //-----------------------------------------------------------------------------
  static void TestNormalise()
  {
    // Implictly testing GetLength().
    mitk::Point3D a;
    a[0] = 1; a[1] = 1; a[2] = 1;

    for (int i = 1; i < 9; i++)
    {
      for (int j = 0; j < 3; j++)
      {
        a[j] = i*j;

        mitk::Point3D b = a;
        mitk::Normalise(b);
        double length = mitk::Length(b);
        MITK_TEST_CONDITION_REQUIRED(mitk::Equal(length, 1),".. Testing TestNormalise length=1 (27 tests)");

        for (int k = -1; k <= 1; k++)
        {
        }
      }
    }

    // Also, if a has zero length, normalise should leave it.
    a[0] = 0; a[1] = 0; a[2] = 0;
    double length = mitk::Length(a);
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(length, 0),".. Testing TestNormalise length=0 zero length vector has length of 1");
    mitk::Normalise(a);
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(a[0], 0),".. Testing TestNormalise a[0] = 0");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(a[1], 0),".. Testing TestNormalise a[1] = 0");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(a[2], 0),".. Testing TestNormalise a[2] = 0");
  }

};

/**
 * Basic test harness for mitkPointUtilsTest
 */
int mitkPointUtilsTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkPointUtilsTest");

  mitkPointUtilsTestClass::TestCalculateStepSize();
  mitkPointUtilsTestClass::TestAreDifferent();
  mitkPointUtilsTestClass::TestGetSquaredDistanceBetweenPoints();
  mitkPointUtilsTestClass::TestGetDifference();
  mitkPointUtilsTestClass::TestNormalise();

  MITK_TEST_END();
}


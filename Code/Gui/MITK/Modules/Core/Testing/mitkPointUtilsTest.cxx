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
#include <cstdlib>
#include <mitkTestingMacros.h>
#include <mitkVector.h>
#include <mitkPointUtils.h>

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

        float distance =  (a[0]-b[0])*(a[0]-b[0])
                        + (a[1]-b[1])*(a[1]-b[1])
                        + (a[2]-b[2])*(a[2]-b[2]);
        float result = mitk::GetSquaredDistanceBetweenPoints(a,b);
        MITK_TEST_CONDITION_REQUIRED(mitk::Equal(result, distance, 0.001),".. Testing GetSquaredDistanceBetweenPoints");
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
    MITK_TEST_OUTPUT(<< "Starting TestNormalise...");

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

    MITK_TEST_OUTPUT(<< "Finished TestNormalise...");
  }


  //-----------------------------------------------------------------------------
  static void TestFindLargestDistanceBetweenTwoPoints()
  {
    MITK_TEST_OUTPUT(<< "Starting TestFindLargestDistanceBetweenTwoPoints...");

    mitk::PointSet::Pointer pointSet = mitk::PointSet::New();

    mitk::Point3D point;
    point[0] = 0;
    point[1] = 0;
    point[2] = 0;
    pointSet->InsertPoint(0, point);
    point[0] = 1;
    pointSet->InsertPoint(1, point);
    point[1] = 2;
    pointSet->InsertPoint(2, point);
    point[2] = 3;
    pointSet->InsertPoint(3, point);

    double distance = mitk::FindLargestDistanceBetweenTwoPoints(*pointSet);
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(distance, 3.741657387, 0.001),".. Testing distance = 3.741657387, with tolerance 0.001, when it actually equals " << distance);

    MITK_TEST_OUTPUT(<< "Finished TestFindLargestDistanceBetweenTwoPoints...");
  }


  //-----------------------------------------------------------------------------
  static void TestScalePointSets()
  {
    MITK_TEST_OUTPUT(<< "Starting TestScalePointSets...");

    mitk::PointSet::Pointer inputPointSet = mitk::PointSet::New();
    mitk::PointSet::Pointer outputPointSet = mitk::PointSet::New();

    mitk::Point3D point;
    point[0] = 1;
    point[1] = 1;
    point[2] = 1;
    inputPointSet->InsertPoint(0, point);

    mitk::ScalePointSets(*inputPointSet, *outputPointSet, 1);
    MITK_TEST_CONDITION_REQUIRED(!mitk::AreDifferent(inputPointSet->GetPoint(0), outputPointSet->GetPoint(0)),".. Testing that scale factor 1 does nothing. " );
    mitk::ScalePointSets(*inputPointSet, *outputPointSet, 2);
    double squaredDistance = mitk::GetSquaredDistanceBetweenPoints(inputPointSet->GetPoint(0), outputPointSet->GetPoint(0));
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(squaredDistance, 3, 0.001),".. Testing squared distance = 3, with tolerance 0.001, when it actually equals " << squaredDistance);
    mitk::ScalePointSets(*inputPointSet, *outputPointSet, 3);
    squaredDistance = mitk::GetSquaredDistanceBetweenPoints(inputPointSet->GetPoint(0), outputPointSet->GetPoint(0));
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(squaredDistance, 12, 0.001),".. Testing squared distance = 12, with tolerance 0.001, when it actually equals " << squaredDistance);

    MITK_TEST_OUTPUT(<< "Finished TestScalePointSets...");
  }


  //-----------------------------------------------------------------------------
  static void TestComputeNormalFromPoints()
  {
    MITK_TEST_OUTPUT(<< "Starting TestComputeNormalFromPoints...");

    // Implicitly tests CrossProduct and CopyValues
    mitk::Point3D a, b, c, output;
    a[0] = 0;
    a[1] = 0;
    a[2] = 0;
    b[0] = 2;
    b[1] = 0;
    b[2] = 0;
    c[0] = 2;
    c[1] = 2;
    c[2] = 0;
    mitk::ComputeNormalFromPoints(a, b, c, output);
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(output[0], 0),".. Testing TestComputeNormalFromPoints output[0] = 0, and it equals:" << output[0]);
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(output[1], 0),".. Testing TestComputeNormalFromPoints output[1] = 0, and it equals:" << output[1]);
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(output[2], -1),".. Testing TestComputeNormalFromPoints output[2] = 1, and it equals:" << output[2]);

    MITK_TEST_OUTPUT(<< "Finished TestComputeNormalFromPoints...");
  }


  //-----------------------------------------------------------------------------
  static void TestFilterMatchingPoints()
  {
    MITK_TEST_OUTPUT(<< "Starting TestFilterMatchingPoints...");

    mitk::PointSet::Pointer fixedPoints = mitk::PointSet::New();
    mitk::PointSet::Pointer movingPoints = mitk::PointSet::New();

    mitk::Point3D p1;
    mitk::Point3D p2;
    mitk::Point3D p3;

    p1[0] = 0;
    p1[1] = 1;
    p1[2] = 2;

    p2[0] = 3;
    p2[1] = 4;
    p2[2] = 5;

    p3[0] = 6;
    p3[1] = 7;
    p3[2] = 8;

    fixedPoints->InsertPoint(1, p1);
    fixedPoints->InsertPoint(2, p2);
    movingPoints->InsertPoint(1, p1);
    movingPoints->InsertPoint(3, p3);

    mitk::PointSet::Pointer outputFixedPoints = mitk::PointSet::New();
    mitk::PointSet::Pointer outputMovingPoints = mitk::PointSet::New();
    int matchedPoints = mitk::FilterMatchingPoints(*fixedPoints, *movingPoints, *outputFixedPoints, *outputMovingPoints);

    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(outputFixedPoints->GetSize(), 1),".. Testing output fixed points has size=1, and it has:" << outputFixedPoints->GetSize());
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(outputMovingPoints->GetSize(), 1),".. Testing output fixed points has size=1, and it has:" << outputMovingPoints->GetSize());
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(matchedPoints, 1),".. Testing output fixed points reports size=1, and it has matchedPoints:" << matchedPoints);

    mitk::Point3D p4 = outputFixedPoints->GetPoint(1);
    mitk::Point3D p5 = outputMovingPoints->GetPoint(1);

    for (int i = 0; i < 3; i++)
    {
      MITK_TEST_CONDITION_REQUIRED(mitk::Equal(p4[i], p5[i]),".. Testing outputs equal,  p4[" << i << "]==" << p4[i] << ", p5[" << i << "]=" << p5[i]);
    }

    MITK_TEST_OUTPUT(<< "Finished TestFilterMatchingPoints...");
  }
  //-----------------------------------------------------------------------------
  static void TestRemoveNaNPoints()
  {
    MITK_TEST_OUTPUT(<< "Starting TestRemoveNaNPoints...");

    mitk::PointSet::Pointer naNPoints = mitk::PointSet::New();
    mitk::PointSet::Pointer noNaNPoints = mitk::PointSet::New();

    mitk::Point3D p1;
    mitk::Point3D p2;
    mitk::Point3D p3;

    p1[0] = 0;
    p1[1] = 1;
    p1[2] = 2;

    p2[0] = 3;
    p2[1] = std::numeric_limits<double>::quiet_NaN();
    p2[2] = 5;

    p3[0] = 6;
    p3[1] = 7;
    p3[2] = 8;

    naNPoints->InsertPoint(1, p1);
    naNPoints->InsertPoint(2, p2);
    noNaNPoints->InsertPoint(1, p1);
    noNaNPoints->InsertPoint(3, p3);

    mitk::PointSet::Pointer outputNaNPoints = mitk::PointSet::New();
    mitk::PointSet::Pointer outputNoNaNPoints = mitk::PointSet::New();
    int naNRemovedPoints = mitk::RemoveNaNPoints(*naNPoints, *outputNaNPoints);
    int noNaNRemovedPoints = mitk::RemoveNaNPoints(*noNaNPoints, *outputNoNaNPoints);

    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(outputNaNPoints->GetSize(), 1),".. Testing output NaN points has size=1, and it has:" << outputNaNPoints->GetSize());
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(outputNoNaNPoints->GetSize(), 2),".. Testing output fixed points has size=2, and it has:" << outputNoNaNPoints->GetSize());
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(naNRemovedPoints, 1),".. Testing 1 NaN point removed");
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(noNaNRemovedPoints, 0),".. Testing 0 non NaN points removed");

    MITK_TEST_OUTPUT(<< "Finished TestRemoveNaNPoints...");
  }


  //-----------------------------------------------------------------------------
  static void TestCheckForNaNPoint()
  {
    MITK_TEST_OUTPUT(<< "Starting TestCheckForNaNPoint...");

    mitk::Point3D p1;
    mitk::Point3D p2;
    mitk::Point3D p3;
    mitk::Point3D p4;
    mitk::Point3D p5;
    mitk::Point3D p6;

    p1[0] = 0;
    p1[1] = 1;
    p1[2] = 2;

    p2[0] = std::numeric_limits<double>::quiet_NaN();
    p2[1] = 4;
    p2[2] = 5;

    p3[0] = std::numeric_limits<double>::infinity();
    p3[1] = 7;
    p3[2] = 8;
    
    p4[0] = 3; 
    p4[1] = std::numeric_limits<float>::quiet_NaN();
    p4[2] = 5;

    p5[0] = 3; 
    p5[1] = 4;
    p5[2] = std::numeric_limits<double>::quiet_NaN();

    p6[0] = std::numeric_limits<double>::quiet_NaN(); 
    p6[1] = std::numeric_limits<double>::quiet_NaN(); 
    p6[2] = std::numeric_limits<double>::quiet_NaN(); 

    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(mitk::CheckForNaNPoint(p1), false),".. Testing that CheckFormNaNPoint returns false for a point with no NaNs");

    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(mitk::CheckForNaNPoint(p2), true),".. Testing that CheckFormNaNPoint returns true for a point with double NaN x");

    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(mitk::CheckForNaNPoint(p4), true),".. Testing that CheckFormNaNPoint returns true for a point with float NaN y");

    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(mitk::CheckForNaNPoint(p5), true),".. Testing that CheckFormNaNPoint returns true for a point with double NaN z");

    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(mitk::CheckForNaNPoint(p3), false),".. Testing that CheckFormNaNPoint returns false for a point with infinite x");

    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(mitk::CheckForNaNPoint(p6), true),".. Testing that CheckFormNaNPoint returns true for a point with all NaNs");

    MITK_TEST_OUTPUT(<< "Finished TestCheckForNaNPoint...");
  }

  //-----------------------------------------------------------------------------
  static void TestRMS()
  {
    MITK_TEST_OUTPUT(<< "Starting TestRMS...");
    
    mitk::PointSet::Pointer fixedPoints = mitk::PointSet::New();
    mitk::PointSet::Pointer movingPoints = mitk::PointSet::New();

    mitk::Point3D p1f, p1m;
    mitk::Point3D p2f, p2m;
    mitk::Point3D p3f, p3m;

    p1f[0] = 0;  p1m[0] = 1;
    p1f[1] = 1;  p1m[1] = 2;
    p1f[2] = 2;  p1m[2] = 3;

    p2f[0] = 3;  p2m[0] = 4;
    p2f[1] = 4;  p2m[1] = 5;
    p2f[2] = 5;  p2m[2] = 6;

    p3f[0] = 6;  p3m[0] = 8;
    p3f[1] = 7;  p3m[1] = 9;
    p3f[2] = 8;  p3m[2] = 10;
    
    fixedPoints->InsertPoint(1, p1f);
    fixedPoints->InsertPoint(2, p2f);
    movingPoints->InsertPoint(1, p1m);
    movingPoints->InsertPoint(3, p3m);

    // RMS should be difference between fixed and moving point 1.
    double rms = mitk::GetRMSErrorBetweenPoints(*fixedPoints, *movingPoints);
    double expected = 1.732050808;
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(rms, expected, 0.00001),".. Testing GetRMSErrorBetweenPoints 1, expected=" << expected << ", actual=" << rms);

    // 2 points with same error gives same RMS.
    movingPoints->InsertPoint(2, p2m);
    rms = mitk::GetRMSErrorBetweenPoints(*fixedPoints, *movingPoints);    
    expected = 1.732050808;
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(rms, expected, 0.00001),".. Testing GetRMSErrorBetweenPoints 2, expected=" << expected << ", actual=" << rms);

    // Adding extra point, which has larger error.
    fixedPoints->InsertPoint(3, p3f);
    rms = mitk::GetRMSErrorBetweenPoints(*fixedPoints, *movingPoints);
    expected = 2.449489743;
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(rms, expected, 0.00001),".. Testing GetRMSErrorBetweenPoints 3, expected=" << expected << ", actual=" << rms);
    
    // Add transformation of -1, -1, -1, which should remove all error, except point 3.
    // And in this case, as it is RMS, we have 3 points, so the mean goes down to 1, and is not SQRT(3).
    mitk::Point3D trans;
    trans[0] = -1;
    trans[1] = -1;
    trans[2] = -1;
    mitk::CoordinateAxesData::Pointer transform = mitk::CoordinateAxesData::New();
    transform->SetTranslation(trans);
    rms = mitk::GetRMSErrorBetweenPoints(*fixedPoints, *movingPoints, transform.GetPointer());
    expected = 1;
    MITK_TEST_CONDITION_REQUIRED(mitk::Equal(rms, expected, 0.00001),".. Testing GetRMSErrorBetweenPoints 4, expected=" << expected << ", actual=" << rms);
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
  mitkPointUtilsTestClass::TestComputeNormalFromPoints();
  mitkPointUtilsTestClass::TestFilterMatchingPoints();
  mitkPointUtilsTestClass::TestRMS();
  mitkPointUtilsTestClass::TestRemoveNaNPoints();
  mitkPointUtilsTestClass::TestCheckForNaNPoint();
  mitkPointUtilsTestClass::TestFindLargestDistanceBetweenTwoPoints();
  mitkPointUtilsTestClass::TestScalePointSets();

  MITK_TEST_END();
}


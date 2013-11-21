/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <random>
#include <mitkTestingMacros.h>
#include <PointClouds/FitPlaneToPointCloudWrapper.h>


//-----------------------------------------------------------------------------
void TestErrorConditions()
{

  try
  {
    // we expect an exception if file name is empty.
    niftk::FitPlaneToPointCloudWrapper::Pointer   fitter = niftk::FitPlaneToPointCloudWrapper::New();
    fitter->FitPlane("");

    MITK_TEST_CONDITION(!"No exception thrown", "FitPlaneToPointCloud: Exception on empty file name");
  }
  catch (const std::runtime_error& e)
  {
    MITK_TEST_CONDITION("Threw and caught correct exception", "FitPlaneToPointCloud: Exception on empty file name");
  }
  catch (...)
  {
    MITK_TEST_CONDITION(!"Threw wrong exception", "FitPlaneToPointCloud: Exception on empty file name");
  }


  try
  {
    // we expect an exception if file does not exist.
    niftk::FitPlaneToPointCloudWrapper::Pointer   fitter = niftk::FitPlaneToPointCloudWrapper::New();
    // lets hope this file does not exist!
    fitter->FitPlane("/tmp/FitPlaneToPointCloudTest.ju7y6tgvbnji87654ertyuhjbv");

    MITK_TEST_CONDITION(!"No exception thrown", "FitPlaneToPointCloud: Exception on non-existent file");
  }
  catch (const std::runtime_error& e)
  {
    MITK_TEST_CONDITION("Threw and caught correct exception", "FitPlaneToPointCloud: Exception on non-existent file");
  }
  catch (...)
  {
    MITK_TEST_CONDITION(!"Threw wrong exception", "FitPlaneToPointCloud: Exception on non-existent file");
  }


  try
  {
    // we expect an exception if we try to get parameters but we never put any data in.
    niftk::FitPlaneToPointCloudWrapper::Pointer   fitter = niftk::FitPlaneToPointCloudWrapper::New();
    float   a, b, c, d;
    fitter->GetParameters(a, b, c, d);

    MITK_TEST_CONDITION(!"No exception thrown", "FitPlaneToPointCloud: Exception on not fitting any data");
  }
  catch (const std::logic_error& e)
  {
    MITK_TEST_CONDITION("Threw and caught correct exception", "FitPlaneToPointCloud: Exception on not fitting any data");
  }
  catch (...)
  {
    MITK_TEST_CONDITION(!"Threw wrong exception", "FitPlaneToPointCloud: Exception on not fitting any data");
  }


  try
  {
    // we expect an exception if we pass in an empty point cloud.
    mitk::PointSet::Pointer   empty = mitk::PointSet::New();
    niftk::FitPlaneToPointCloudWrapper::Pointer   fitter = niftk::FitPlaneToPointCloudWrapper::New();
    fitter->FitPlane(empty);

    MITK_TEST_CONDITION(!"No exception thrown", "FitPlaneToPointCloud: Exception on empty point cloud");
  }
  catch (const std::runtime_error& e)
  {
    MITK_TEST_CONDITION("Threw and caught correct exception", "FitPlaneToPointCloud: Exception on empty point cloud");
  }
  catch (...)
  {
    MITK_TEST_CONDITION(!"Threw wrong exception", "FitPlaneToPointCloud: Exception on empty point cloud");
  }


  try
  {
    // we expect NO exception if we pass in a point cloud with some (well-formed) data.
    mitk::PointSet::Pointer   stuff = mitk::PointSet::New();
    float   points[] =
    {
      0, 0, 0,
      1, 0, 0,
      0, 1, 0,
      1, 1, 0
    };
    for (int i = 0; i < (sizeof(points) / sizeof(points[0])); i += 3)
      stuff->InsertPoint(i, mitk::PointSet::PointType(&points[i]));

    niftk::FitPlaneToPointCloudWrapper::Pointer   fitter = niftk::FitPlaneToPointCloudWrapper::New();
    fitter->FitPlane(stuff);

    float   a, b, c, d;
    fitter->GetParameters(a, b, c, d);

    MITK_TEST_CONDITION("No exception thrown", "FitPlaneToPointCloud: No exception on valid point cloud");
  }
  catch (...)
  {
    MITK_TEST_CONDITION(!"Threw an exception", "FitPlaneToPointCloud: No exception on valid point cloud");
  }

}


//-----------------------------------------------------------------------------
void TestSimpleCases()
{
  float   a, b, c, d;

  mitk::PointSet::Pointer   xyplane = mitk::PointSet::New();
  for (int i = 0; i < 100; ++i)
  {
    float   p[3] = {std::rand(), std::rand(), -2};
    xyplane->InsertPoint(i, mitk::PointSet::PointType(&p[0]));
  }

  niftk::FitPlaneToPointCloudWrapper::Pointer   fitter = niftk::FitPlaneToPointCloudWrapper::New();
  fitter->FitPlane(xyplane);
  fitter->GetParameters(a, b, c, d);
  MITK_TEST_CONDITION(std::abs(0 - a) < 0.01f, "FitPlaneToPointCloud unit test xy-plane: Parameter A in range");
  MITK_TEST_CONDITION(std::abs(0 - b) < 0.01f, "FitPlaneToPointCloud unit test xy-plane: Parameter B in range");
  MITK_TEST_CONDITION(std::abs(1 - c) < 0.01f, "FitPlaneToPointCloud unit test xy-plane: Parameter C in range");
  MITK_TEST_CONDITION(std::abs(2 - d) < 0.01f, "FitPlaneToPointCloud unit test xy-plane: Parameter D in range");


  mitk::PointSet::Pointer   yzplane = mitk::PointSet::New();
  for (int i = 0; i < 100; ++i)
  {
    float   p[3] = {0, std::rand(), std::rand()};
    yzplane->InsertPoint(i, mitk::PointSet::PointType(&p[0]));
  }

  fitter->FitPlane(yzplane);
  fitter->GetParameters(a, b, c, d);
  MITK_TEST_CONDITION(std::abs(1 - a) < 0.01f, "FitPlaneToPointCloud unit test yz-plane: Parameter A in range");
  MITK_TEST_CONDITION(std::abs(0 - b) < 0.01f, "FitPlaneToPointCloud unit test yz-plane: Parameter B in range");
  MITK_TEST_CONDITION(std::abs(0 - c) < 0.01f, "FitPlaneToPointCloud unit test yz-plane: Parameter C in range");
  MITK_TEST_CONDITION(std::abs(0 - d) < 0.01f, "FitPlaneToPointCloud unit test yz-plane: Parameter D in range");

}


//-----------------------------------------------------------------------------
void RunRegressionTest(const char* filename)
{
  // no additional try/catch here. we expect this to succeed!

  niftk::FitPlaneToPointCloudWrapper::Pointer   fitter = niftk::FitPlaneToPointCloudWrapper::New();
  fitter->FitPlane(filename);

  float   a, b, c, d;
  fitter->GetParameters(a, b, c, d);

  // not sure how accurate we expect this to be...
  // plane fit involves random sampling so might return different output each time.
  // the numbers are from actual output but i rounded them to 2 digits.
  MITK_TEST_CONDITION(std::abs( -0.03f - a) < 0.01f, "FitPlaneToPointCloud regression test: Parameter A in range");
  MITK_TEST_CONDITION(std::abs( -0.50f - b) < 0.01f, "FitPlaneToPointCloud regression test: Parameter B in range");
  MITK_TEST_CONDITION(std::abs(  0.86f - c) < 0.01f, "FitPlaneToPointCloud regression test: Parameter C in range");
  MITK_TEST_CONDITION(std::abs(-35.86f - d) < 0.01f, "FitPlaneToPointCloud regression test: Parameter D in range");
}


//-----------------------------------------------------------------------------
int FitPlaneToPointCloudTest(int argc, char* argv[])
{
  if (argc != 2)
  {
    std::cerr << "Expecting file name as a parameter" << std::endl;
    return EXIT_FAILURE;
  }

  TestErrorConditions();
  TestSimpleCases();
  RunRegressionTest(argv[1]);

  return EXIT_SUCCESS;
}

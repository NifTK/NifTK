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
#include <PointClouds/mitkMergePointClouds.h>


//-----------------------------------------------------------------------------
static void TestErrorConditions()
{

  try
  {
    // we expect an exception if file name is empty.
    mitk::MergePointClouds::Pointer   merger = mitk::MergePointClouds::New();
    merger->AddPointSet("");

    MITK_TEST_CONDITION(!"No exception thrown", "MergePointClouds: Exception on empty file name");
  }
  catch (const std::runtime_error& e)
  {
    MITK_TEST_CONDITION("Threw and caught correct exception", "MergePointClouds: Exception on empty file name");
  }
  catch (...)
  {
    MITK_TEST_CONDITION(!"Threw wrong exception", "MergePointClouds: Exception on empty file name");
  }


  try
  {
    // we expect an exception if file does not exist.
    mitk::MergePointClouds::Pointer   merger = mitk::MergePointClouds::New();
    // lets hope this file does not exist!
    merger->AddPointSet("/tmp/MergePointClouds.ju7y6tgvbnji87654ertyuhjbv");

    MITK_TEST_CONDITION(!"No exception thrown", "MergePointClouds: Exception on non-existent file");
  }
  catch (const std::runtime_error& e)
  {
    MITK_TEST_CONDITION("Threw and caught correct exception", "MergePointClouds: Exception on non-existent file");
  }
  catch (...)
  {
    MITK_TEST_CONDITION(!"Threw wrong exception", "MergePointClouds: Exception on non-existent file");
  }


  try
  {
    // we expect no exception getting output without putting anything in first.
    // output should simply be empty.
    mitk::MergePointClouds::Pointer   merger = mitk::MergePointClouds::New();
    mitk::PointSet::Pointer   empty = merger->GetOutput();

    MITK_TEST_CONDITION("No exception thrown", "MergePointClouds: No exception on not adding in any data");
  }
  catch (...)
  {
    MITK_TEST_CONDITION(!"Threw an exception", "MergePointClouds: No exception on not adding in any data");
  }


  try
  {
    // we expect an exception if we pass in a null point cloud.
    mitk::PointSet::Pointer   null;
    mitk::MergePointClouds::Pointer   merger = mitk::MergePointClouds::New();
    merger->AddPointSet(null);

    MITK_TEST_CONDITION(!"No exception thrown", "MergePointClouds: Exception on null point cloud");
  }
  catch (const std::runtime_error& e)
  {
    MITK_TEST_CONDITION("Threw and caught correct exception", "MergePointClouds: Exception on null point cloud");
  }
  catch (...)
  {
    MITK_TEST_CONDITION(!"Threw wrong exception", "MergePointClouds: Exception on null point cloud");
  }
}


//-----------------------------------------------------------------------------
static void TestSimpleCases()
{
  mitk::PointSet::Pointer   one = mitk::PointSet::New();
  {
    float   p[3] = {std::rand(), std::rand(), -2};
    one->InsertPoint(3, mitk::PointSet::PointType(&p[0]));
  }

  mitk::MergePointClouds::Pointer   merger = mitk::MergePointClouds::New();
  merger->AddPointSet(one);
  mitk::PointSet::Pointer   merged = merger->GetOutput();

  MITK_TEST_CONDITION(merged->GetSize() == one->GetSize(), "MergePointClouds unit test single-point point set: output has correct size");
  MITK_TEST_CONDITION(std::abs(merged->Begin()->Value()[0] - one->Begin()->Value()[0]) < 0.001f, "MergePointClouds unit test single-point point set: coordinate X matches");
  MITK_TEST_CONDITION(std::abs(merged->Begin()->Value()[1] - one->Begin()->Value()[1]) < 0.001f, "MergePointClouds unit test single-point point set: coordinate Y matches");
  MITK_TEST_CONDITION(std::abs(merged->Begin()->Value()[2] - one->Begin()->Value()[2]) < 0.001f, "MergePointClouds unit test single-point point set: coordinate Z matches");


  mitk::PointSet::Pointer   two = mitk::PointSet::New();
  for (int i = 0; i < 2; ++i)
  {
    float   p[3] = {0, std::rand(), std::rand()};
    two->InsertPoint(i, mitk::PointSet::PointType(&p[0]));
  }

  merger->AddPointSet(two);
  // after adding another point set, the output returned previously should not have been modified!
  MITK_TEST_CONDITION(merged->GetSize() == one->GetSize(), "MergePointClouds unit test single-point point set: previous output has correct size");

  merged = merger->GetOutput();
  MITK_TEST_CONDITION(merged->GetSize() == (one->GetSize() + two->GetSize()), "MergePointClouds unit test merge two point sets: output has correct size");
  // dont check for point coordinates. order of iterating points might have changed.
}


//-----------------------------------------------------------------------------
int mitkMergePointCloudsTest(int argc, char* argv[])
{
  TestErrorConditions();
  TestSimpleCases();

  return EXIT_SUCCESS;
}

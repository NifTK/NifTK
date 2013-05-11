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

#include <vtkMatrix4x4.h>
#include <mitkTestingMacros.h>
#include <mitkCoordinateAxesData.h>

/**
 * \class CoordinateAxesDataTest
 * \brief Test class for mitk::CoordinateAxesData.
 */
class CoordinateAxesDataTest
{

public:

  //-----------------------------------------------------------------------------
  static void TestSetGetMatrix()
  {
    MITK_TEST_OUTPUT(<< "Starting TestSetGetMatrix...");

    vtkMatrix4x4 *input = vtkMatrix4x4::New();
    vtkMatrix4x4 *output = vtkMatrix4x4::New();

    mitk::CoordinateAxesData::Pointer coordinateAxes = mitk::CoordinateAxesData::New();
    input->Identity();

    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 4; j++)
      {
        input->SetElement(i, j, i*4 + j);
      }
    }

    coordinateAxes->SetVtkMatrix(*input);
    coordinateAxes->GetVtkMatrix(*output);

    for (int i = 0; i < 3; i++)
    {
      for (int j = 0; j < 4; j++)
      {
        MITK_TEST_CONDITION_REQUIRED(mitk::Equal(input->GetElement(i, j), output->GetElement(i, j)),".. Testing Equality of matrices i=" << i << ", j=" << j);
      }
    }

    MITK_TEST_OUTPUT(<< "Finished TestSetGetMatrix...");
  }
}; // end class


/**
 * \file Test harness for mitk::CoordinateAxesData.
 */
int mitkCoordinateAxesDataTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkCoordinateAxesDataTest");

  CoordinateAxesDataTest::TestSetGetMatrix();

  MITK_TEST_END();
}


/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <cstdlib>
#include <mitkTestingMacros.h>
#include <mitkPointSet.h>
#include <mitkPointBasedRegistration.h>
#include <vtkMatrix4x4.h>
#include <vtkSmartPointer.h>

/**
 * \file mitkPointBasedRegistrationTest.cxx
 * \brief Tests for mitk::PointBasedRegistration.
 */
int mitkPointBasedRegistrationTest(int /*argc*/, char* /*argv*/[])
{
  MITK_TEST_OUTPUT(<< "Started mitkPointBasedRegistrationTest...");

  mitk::PointSet::Pointer fixedPoints = mitk::PointSet::New();
  mitk::PointSet::Pointer movingPoints = mitk::PointSet::New();

  mitk::Point3D f1;
  mitk::Point3D f2;
  mitk::Point3D f3;
  mitk::Point3D f4;
  mitk::Point3D f5;
  mitk::Point3D f6;

  mitk::Point3D m1;
  mitk::Point3D m2;
  mitk::Point3D m3;
  mitk::Point3D m4;
  mitk::Point3D m5;
  mitk::Point3D m6;

  f1[0] = 0; f1[1] = 0; f1[2] = 0;
  f2[0] = 1; f2[1] = 0; f2[2] = 0;
  f3[0] = 0; f3[1] = 1; f3[2] = 0;
  f4[0] = 0; f4[1] = 0; f4[2] = 1;

  m1[0] = 1; m1[1] = 0; m1[2] = 0;
  m2[0] = 2; m2[1] = 0; m2[2] = 0;
  m3[0] = 1; m3[1] = 1; m3[2] = 0;
  m4[0] = 1; m4[1] = 0; m4[2] = 1;

  fixedPoints->InsertPoint(1, f1);
  fixedPoints->InsertPoint(2, f2);
  fixedPoints->InsertPoint(3, f3);
  fixedPoints->InsertPoint(4, f4);

  movingPoints->InsertPoint(1, m1);
  movingPoints->InsertPoint(2, m2);
  movingPoints->InsertPoint(3, m3);
  movingPoints->InsertPoint(4, m4);

  vtkSmartPointer<vtkMatrix4x4> matrix = vtkMatrix4x4::New();
  matrix->Identity();
  double fre = 0;
  double fre2 = 0;

  mitk::PointBasedRegistration::Pointer registration = mitk::PointBasedRegistration::New();
  registration->SetUseICPInitialisation(false);
  registration->SetUsePointIDToMatchPoints(false);
  fre = registration->Update(fixedPoints, movingPoints, *matrix);

  MITK_TEST_CONDITION_REQUIRED(mitk::Equal(fre, 0),".. Testing fre=0, and it equals:" << fre);
  MITK_TEST_CONDITION_REQUIRED(mitk::Equal(matrix->GetElement(0,3), -1),".. Testing x translation=-1 and it equals:" << matrix->GetElement(0,3));

  f5[0] = 6; f5[1] = 7; f5[2] = 8;
  m5[0] = 3; m5[1] = 5; m5[2] = 9;

  fixedPoints->InsertPoint(6, f5);  // PointID does not have to be contiguous.
  movingPoints->InsertPoint(7, m5); // We just need a different ID for each of these points
  registration->SetUseICPInitialisation(false);
  registration->SetUsePointIDToMatchPoints(true);
  fre2 = registration->Update(fixedPoints, movingPoints, *matrix);

  MITK_TEST_CONDITION_REQUIRED(mitk::Equal(fre, fre2),".. Testing fre==fre2,and fre=" << fre << ", fre2=" << fre2);
  MITK_TEST_CONDITION_REQUIRED(mitk::Equal(matrix->GetElement(0,3), -1),".. Testing x translation=-1 and it equals:" << matrix->GetElement(0,3));

  MITK_TEST_OUTPUT(<< "Finished mitkPointBasedRegistrationTest...");

  return EXIT_SUCCESS;
}



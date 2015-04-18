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

#include <niftkArunLeastSquaresPointRegistration.h>
#include <mitkTestingMacros.h>
#include <mitkLogMacros.h>
#include <mitkPointSet.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>

void TranslationTest()
{
  mitk::Point3D f1;
  mitk::Point3D f2;
  mitk::Point3D f3;
  mitk::Point3D f4;

  mitk::Point3D m1;
  mitk::Point3D m2;
  mitk::Point3D m3;
  mitk::Point3D m4;

  f1[0] = 0; f1[1] = 0; f1[2] = 0;
  f2[0] = 1; f2[1] = 0; f2[2] = 0;
  f3[0] = 0; f3[1] = 1; f3[2] = 0;
  f4[0] = 0; f4[1] = 0; f4[2] = 1;

  m1[0] = 1; m1[1] = 0; m1[2] = 0;
  m2[0] = 2; m2[1] = 0; m2[2] = 0;
  m3[0] = 1; m3[1] = 1; m3[2] = 0;
  m4[0] = 1; m4[1] = 0; m4[2] = 1;

  mitk::PointSet::Pointer fixedPoints = mitk::PointSet::New();
  mitk::PointSet::Pointer movingPoints = mitk::PointSet::New();

  fixedPoints->InsertPoint(1, f1);
  fixedPoints->InsertPoint(2, f2);
  fixedPoints->InsertPoint(3, f3);
  fixedPoints->InsertPoint(4, f4);

  movingPoints->InsertPoint(1, m1);
  movingPoints->InsertPoint(2, m2);
  movingPoints->InsertPoint(3, m3);
  movingPoints->InsertPoint(4, m4);

  vtkSmartPointer<vtkMatrix4x4> matrix = vtkSmartPointer<vtkMatrix4x4>::New();
  matrix->Identity();

  double fre = niftk::PointBasedRegistrationUsingSVD(fixedPoints, movingPoints, *matrix);
  MITK_TEST_CONDITION_REQUIRED(mitk::Equal(fre, 0),".. Testing fre=0, and it equals:" << fre);
  MITK_TEST_CONDITION_REQUIRED(mitk::Equal(matrix->GetElement(0,3), -1),".. Testing x translation=-1 and it equals:" << matrix->GetElement(0,3));
}

void RotationTest()
{
  mitk::Point3D f1;
  mitk::Point3D f2;
  mitk::Point3D f3;
  mitk::Point3D f4;
  mitk::Point3D f5;

  mitk::Point3D m1;
  mitk::Point3D m2;
  mitk::Point3D m3;
  mitk::Point3D m4;
  mitk::Point3D m5;

  f1[0] = 39.3047;
  f1[1] = 71.7057;
  f1[2] = 372.7200;
  f2[0] = 171.8440;
  f2[1] = 65.8063;
  f2[2] = 376.3250;
  f3[0] = 312.0440;
  f3[1] = 77.5614;
  f3[2] = 196.0000;
  f4[0] = 176.5280;
  f4[1] = 78.7922;
  f4[2] = 8.0000;
  f5[0] = 43.4659;
  f5[1] = 53.2688;
  f5[2] = 10.5000;

  m1[0] = 192.8328;
  m1[1] = 290.2859;
  m1[2] = 155.4227;
  m2[0] = 295.6876;
  m2[1] = 287.9700;
  m2[2] = 71.5776;
  m3[0] = 301.1553;
  m3[1] = 183.6233;
  m3[2] = -131.8756;
  m4[0] = 102.1668;
  m4[1] = 66.2676;
  m4[2] = -150.3493;
  m5[0] = 15.3302;
  m5[1] = 48.0055;
  m5[2] = -47.9328;

  mitk::PointSet::Pointer fixedPoints = mitk::PointSet::New();
  mitk::PointSet::Pointer movingPoints = mitk::PointSet::New();

  fixedPoints->InsertPoint(1, f1);
  fixedPoints->InsertPoint(2, f2);
  fixedPoints->InsertPoint(3, f3);
  fixedPoints->InsertPoint(4, f4);
  fixedPoints->InsertPoint(5, f5);

  movingPoints->InsertPoint(1, m1);
  movingPoints->InsertPoint(2, m2);
  movingPoints->InsertPoint(3, m3);
  movingPoints->InsertPoint(4, m4);
  movingPoints->InsertPoint(5, m5);

  vtkSmartPointer<vtkMatrix4x4> expected = vtkSmartPointer<vtkMatrix4x4>::New();
  expected->Identity();

  expected->SetElement(0, 0, 0.7431);
  expected->SetElement(0, 1, -0.0000);
  expected->SetElement(0, 2, -0.6691);
  expected->SetElement(1, 0, -0.4211);
  expected->SetElement(1, 1, 0.7771);
  expected->SetElement(1, 2, -0.4677 );
  expected->SetElement(2, 0, 0.5200);
  expected->SetElement(2, 1, 0.6293);
  expected->SetElement(2, 2, 0.5775);

  vtkSmartPointer<vtkMatrix4x4> actual = vtkSmartPointer<vtkMatrix4x4>::New();
  actual->Identity();

  double fre = niftk::PointBasedRegistrationUsingSVD(fixedPoints, movingPoints, *actual);

  double tolerance = 0.001;
  MITK_TEST_CONDITION_REQUIRED(fre < tolerance, ".. Testing fre < " << tolerance << " and it equals:" << fre);

  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      MITK_TEST_CONDITION_REQUIRED(fabs(expected->GetElement(i,j) - actual->GetElement(i, j)) < tolerance,
                                   ".. With tolerance " << tolerance
                                   << ", testing " << i
                                   << ", " << j
                                   << "=" << expected->GetElement(i,j)
                                   << ", but got " << actual->GetElement(i,j));
    }
  }

}

int niftkArunSVDRegTest ( int argc, char * argv[] )
{
  // always start with this!
  MITK_TEST_BEGIN("niftkArunSVDRegTest");

  // When tests fail they throw exceptions, so no need to worry about return codes.
  TranslationTest();
  RotationTest();

  // always end with this!
  MITK_TEST_END();
}

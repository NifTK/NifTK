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
#include <mitkExceptionMacro.h>
#include <mitkLogMacros.h>
#include <mitkPointSet.h>
#include <mitkIOUtil.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <niftkFileHelper.h>
#include <mitkFileIOUtils.h>

/**
 * \file niftkArunSVDRegTest.cxx
 * \brief Unit test for niftk::PointBasedRegistrationUsingSVD
 */
int niftkArunSVDRegTest ( int argc, char * argv[] )
{
  // always start with this!
  MITK_TEST_BEGIN("niftkArunSVDRegTest");

  if (argc != 5)
  {
    mitkThrow() << "Usage: niftkArunSVDRegTest fixed.mps moving.mps expected.4x4 tolerance";
  }
  double tolerance = atof(argv[4]);

  mitk::PointSet::Pointer fixedPoints = mitk::IOUtil::LoadPointSet(argv[1]);
  mitk::PointSet::Pointer movingPoints = mitk::IOUtil::LoadPointSet(argv[2]);
  vtkSmartPointer<vtkMatrix4x4> expected = mitk::LoadVtkMatrix4x4FromFile(argv[3]);
  vtkSmartPointer<vtkMatrix4x4> actual = vtkSmartPointer<vtkMatrix4x4>::New();
  double fre = niftk::PointBasedRegistrationUsingSVD(fixedPoints, movingPoints, *actual);

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      MITK_TEST_CONDITION_REQUIRED(fabs(expected->GetElement(i,j) - actual->GetElement(i, j)) < tolerance,
                                   ".. With tolerance " << tolerance
                                   << ", testing " << i
                                   << ", " << j
                                   << "=" << expected->GetElement(i,j)
                                   << ", but got " << actual->GetElement(i,j));
    }
  }
  MITK_TEST_CONDITION_REQUIRED(fre < tolerance, ".. Testing fre < " << tolerance << " and it equals:" << fre);

  // always end with this!
  MITK_TEST_END();
}

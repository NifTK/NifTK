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
 * \file niftkArunSVDExceptionTest.cxx
 * \brief Unit test for error cases in niftk::PointBasedRegistrationUsingSVD
 */
int niftkArunSVDExceptionTest ( int argc, char * argv[] )
{
  // always start with this!
  MITK_TEST_BEGIN("niftkArunSVDExceptionTest");

  if (argc != 3)
  {
    mitkThrow() << "Usage: niftkArunSVDExceptionTest fixed.mps moving.mps";
  }
  mitk::PointSet::Pointer fixedPoints = mitk::IOUtil::LoadPointSet(argv[1]);
  mitk::PointSet::Pointer movingPoints = mitk::IOUtil::LoadPointSet(argv[2]);

  try
  {
    vtkSmartPointer<vtkMatrix4x4> actual = vtkSmartPointer<vtkMatrix4x4>::New();
    niftk::PointBasedRegistrationUsingSVD(fixedPoints, movingPoints, *actual);
    mitkThrow() << "Should have thrown exception";
  }
  catch (const mitk::Exception& e)
  {
    std::cout << "Caught exception e=" << e.what() << std::endl;
  }

  mitkThrow() << "Forcing an exception";

  // always end with this!
  MITK_TEST_END();
}

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

#include <niftkFileHelper.h>
#include <niftkMathsUtils.h>
#include <mitkTestingMacros.h>
#include <mitkExceptionMacro.h>
#include <mitkLogMacros.h>
#include <mitkOpenCVMaths.h>
#include <mitkIOUtil.h>
#include <mitkFileIOUtils.h>
#include <niftkUltrasoundPointerBasedCalibration.h>
#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>
#include <cmath>

/**
 * \file Test harness for Ultrasound Pointer based calibration (Muratore 2001).
 */
int niftkUltrasoundPointerBasedCalibrationTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("niftkUltrasoundPointerBasedCalibrationTest");

  if (argc != 4)
  {
    mitkThrow() << "Usage: niftkUltrasoundPointerBasedCalibrationTest image.mps sensor.mps expected.4x4" << std::endl;
  }
  mitk::PointSet::Pointer imagePoints = mitk::IOUtil::LoadPointSet(argv[1]);
  mitk::PointSet::Pointer sensorPoints = mitk::IOUtil::LoadPointSet(argv[2]);

  vtkSmartPointer<vtkMatrix4x4> expectedResult = mitk::LoadVtkMatrix4x4FromFile(argv[3]);

  niftk::UltrasoundPointerBasedCalibration::Pointer calibrator = niftk::UltrasoundPointerBasedCalibration::New();
  calibrator->SetImagePoints(imagePoints);
  calibrator->SetSensorPoints(sensorPoints);

  double residual = calibrator->DoPointerBasedCalibration();

  vtkSmartPointer<vtkMatrix4x4> actualResult = calibrator->GetCalibrationMatrix();
  mitk::SaveVtkMatrix4x4ToFile("/tmp/matt.4x4", *actualResult);

  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      MITK_TEST_CONDITION_REQUIRED(niftk::IsCloseToZero(
                                     fabs(expectedResult->GetElement(i, j)
                                          - actualResult->GetElement(i, j)), 0.001),
                                   "Checking element " << i << ", " << j << " is correct, expecting "
                                   << expectedResult->GetElement(i,j) << ", and got "
                                   << actualResult->GetElement(i, j));
    }
  }
  MITK_TEST_CONDITION_REQUIRED(residual < 0.1, "Checking residual < 0.001 (as there is no noise), and got " << residual);
  MITK_TEST_END();
}



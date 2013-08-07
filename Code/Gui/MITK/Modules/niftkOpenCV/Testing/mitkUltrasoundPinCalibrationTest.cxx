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

#include <mitkTestingMacros.h>
#include <mitkLogMacros.h>
#include <mitkIOUtil.h>

/**
 * \class UltrasoundPinCalibrationTest
 * \brief Test class for ultrasound pin calibration.
 */
class UltrasoundPinCalibrationTest
{

public:

  //-----------------------------------------------------------------------------
  static void DoCalibration()
  {
    MITK_TEST_OUTPUT(<< "Starting DoCalibration...");

    MITK_TEST_OUTPUT(<< "Finished DoCalibration...");
  }
};

/**
 * \file Test harness for Ultrasound Pin Calibration
 */
int mitkUltrasoundPinCalibrationTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkUltrasoundPinCalibrationTest");

  UltrasoundPinCalibrationTest::UltrasoundPinCalibrationTest();

  MITK_TEST_END();
}

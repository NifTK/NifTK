/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <mitkTestingMacros.h>
#include <mitkLogMacros.h>
#include <strstream>
#include <string>
#include <CameraCalibration/mitkCameraCalibrationFacade.h>


//-----------------------------------------------------------------------------
static void CheckExceptionsForLoadingFromPlaintext()
{
  cv::Mat intr = cvCreateMat(3, 3, CV_64FC1);
  cv::Mat dist = cvCreateMat(1, 4, CV_64FC1);

  {
    std::string   wellformed("1 2 3 4 5 6 7 8 9 10 11 12 13\n");
    std::ofstream  wellformedfile("CheckExceptionsForLoadingFromPlaintext.wellformed.txt");
    wellformedfile << wellformed;
    wellformedfile.close();
  }

  {
    std::string   wellformednoeol("1 2 3 4 5 6 7 8 9 10 11 12 13");
    std::ofstream  wellformednoeolfile("CheckExceptionsForLoadingFromPlaintext.wellformednoeol.txt");
    wellformednoeolfile << wellformednoeol;
    wellformednoeolfile.close();
  }

  {
    std::string   tooshort("1 2 3 4 5 6 7 8 9 10 11 12");
    std::ofstream  tooshortfile("CheckExceptionsForLoadingFromPlaintext.tooshort.txt");
    tooshortfile << tooshort;
    tooshortfile.close();
  }

  {
    std::string   garbage("a 2 3 4 5 6 7 8 9 10 11 12");
    std::ofstream  garbagefile("CheckExceptionsForLoadingFromPlaintext.garbage.txt");
    garbagefile << garbage;
    garbagefile.close();
  }

  try
  {
    mitk::LoadCameraIntrinsicsFromPlainText("CheckExceptionsForLoadingFromPlaintext.wellformed.txt", &intr, &dist);
    MITK_TEST_CONDITION("No exception thrown", "CheckExceptionsForLoadingFromPlaintext: No exception for wellformed file");
  }
  catch (...)
  {
    MITK_TEST_CONDITION(!"Caught exception", "CheckExceptionsForLoadingFromPlaintext: No exception for wellformed file");
  }

  try
  {
    mitk::LoadCameraIntrinsicsFromPlainText("CheckExceptionsForLoadingFromPlaintext.wellformednoeol.txt", &intr, &dist);
    MITK_TEST_CONDITION("No exception thrown", "CheckExceptionsForLoadingFromPlaintext: No exception for wellformed file without final EOL");
  }
  catch (...)
  {
    MITK_TEST_CONDITION(!"Caught exception", "CheckExceptionsForLoadingFromPlaintext: No exception for wellformed file without final EOL");
  }

  try
  {
    mitk::LoadCameraIntrinsicsFromPlainText("CheckExceptionsForLoadingFromPlaintext.tooshort.txt", &intr, &dist);
    MITK_TEST_CONDITION(!"No exception thrown", "CheckExceptionsForLoadingFromPlaintext: Exception for file too short");
  }
  catch (...)
  {
    MITK_TEST_CONDITION("Caught exception", "CheckExceptionsForLoadingFromPlaintext: Exception for file too short");
  }

  try
  {
    mitk::LoadCameraIntrinsicsFromPlainText("CheckExceptionsForLoadingFromPlaintext.garbage.txt", &intr, &dist);
    MITK_TEST_CONDITION(!"No exception thrown", "CheckExceptionsForLoadingFromPlaintext: Exception for file with garbage");
  }
  catch (...)
  {
    MITK_TEST_CONDITION("Caught exception", "CheckExceptionsForLoadingFromPlaintext: Exception for file with garbage");
  }
}


//-----------------------------------------------------------------------------
int mitkCameraCalibrationFacadeTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkCameraCalibrationFacadeTest");

  CheckExceptionsForLoadingFromPlaintext();


  MITK_TEST_END();
}

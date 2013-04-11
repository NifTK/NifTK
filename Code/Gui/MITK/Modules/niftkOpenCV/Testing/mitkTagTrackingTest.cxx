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
#include <cv.h>
#include <highgui.h>
#include "mitkTagTrackingFacade.h"

/**
 * \class TagTrackingTest
 * \brief Test class for Tag Tracking, using data from our Viking 3D Stereo Laparoscope.
 */
class TagTrackingTest
{

public:

  //-----------------------------------------------------------------------------
  static void Test3DReconstruction(const std::string& leftImage,
                                   const std::string& rightImage,
                                   const std::string& leftIntrinsics,
                                   const std::string& rightIntrinsics,
                                   const std::string& rightToLeftRotationVector,
                                   const std::string& rightToLeftTranslationVector
                                          )
  {
    MITK_TEST_OUTPUT(<< "Starting Test3DReconstruction...");

    cv::Mat li = cv::imread(leftImage, CV_LOAD_IMAGE_COLOR);
    if(!li.data )
    {
      MITK_TEST_FAILED_MSG(<< "Could not open or find the image:" << leftImage);
    }
    cv::Mat ri = cv::imread(rightImage, CV_LOAD_IMAGE_COLOR);
    if(!ri.data )
    {
      MITK_TEST_FAILED_MSG(<< "Could not open or find the image:" << rightImage);
    }

    cv::Mat liMat = cv::Mat(3, 3, CV_32FC1);
    cv::FileStorage fsli(leftIntrinsics, cv::FileStorage::READ);
    fsli["calib_txt_left_intrinsic"] >> liMat;
    fsli.release();

    cv::Mat riMat = cv::Mat(3, 3, CV_32FC1);
    cv::FileStorage fsri(rightIntrinsics, cv::FileStorage::READ);
    fsri["calib_txt_right_intrinsic"] >> riMat;
    fsri.release();

    cv::Mat r2lRotVector = cv::Mat(1, 3, CV_32FC1);
    cv::FileStorage fsr2lr(rightToLeftRotationVector, cv::FileStorage::READ);
    fsr2lr["left-1095_png_r2l_rotation"] >> r2lRotVector;
    fsr2lr.release();

    cv::Mat r2lTrans = cv::Mat(1, 3, CV_32FC1);
    cv::FileStorage fsr2lt(rightToLeftTranslationVector, cv::FileStorage::READ);
    fsr2lt["left-1095_png_r2l_translation"] >> r2lTrans;
    fsr2lt.release();

    MITK_TEST_OUTPUT(<< "Loaded data...");
    MITK_TEST_OUTPUT(<< "left intrinsic=" << liMat);
    MITK_TEST_OUTPUT(<< "right intrinsic=" << riMat);
    MITK_TEST_OUTPUT(<< "r2l rot=" << r2lRotVector);
    MITK_TEST_OUTPUT(<< "r2l trans=" << r2lTrans);

    std::map<int, cv::Point3f> result = mitk::DetectMarkerPairs(li, ri, liMat, riMat, r2lRotVector, r2lTrans);

    MITK_TEST_CONDITION_REQUIRED(result.size() == 2,".. Testing we got 2 points out");

    MITK_TEST_OUTPUT(<< "Finished Test3DReconstruction...");
  }

}; // end class


/**
 * \file Test harness for Tracking Tag detection.
 */
int mitkTagTrackingTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkTagTrackingTest");

  std::string leftImage = argv[1];
  std::string rightImage = argv[2];
  std::string leftIntrinsics = argv[3];
  std::string rightIntrinsics = argv[4];
  std::string rightToLeftRotationVector = argv[5];
  std::string rightToLeftTranslationVector = argv[6];

  TagTrackingTest::Test3DReconstruction(leftImage, rightImage, leftIntrinsics, rightIntrinsics, rightToLeftRotationVector, rightToLeftTranslationVector);

  MITK_TEST_END();
}

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
#include <cv.h>
#include <highgui.h>
#include <mitkMonoTagExtractor.h>
#include <mitkStereoTagExtractor.h>

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

    mitk::Image::Pointer leftMitkImage = mitk::IOUtil::LoadImage(leftImage);
    MITK_TEST_CONDITION_REQUIRED(leftMitkImage.IsNotNull(), "Checking leftMitkImage is not null");

    mitk::Image::Pointer rightMitkImage = mitk::IOUtil::LoadImage(rightImage);
    MITK_TEST_CONDITION_REQUIRED(rightMitkImage.IsNotNull(), "Checking rightMitkImage is not null");

    CvMat *leftIntMat = (CvMat*)cvLoad(leftIntrinsics.c_str());
    MITK_TEST_CONDITION_REQUIRED(leftIntMat != NULL, "Checking leftIntMat is not null");

    CvMat *rightIntMat = (CvMat*)cvLoad(rightIntrinsics.c_str());
    MITK_TEST_CONDITION_REQUIRED(rightIntMat != NULL, "Checking rightIntMat is not null");

    CvMat *r2lRotMat = (CvMat*)cvLoad(rightToLeftRotationVector.c_str());
    MITK_TEST_CONDITION_REQUIRED(r2lRotMat != NULL, "Checking r2lRotMat is not null");

    CvMat *r2lTrnMat = (CvMat*)cvLoad(rightToLeftTranslationVector.c_str());
    MITK_TEST_CONDITION_REQUIRED(r2lTrnMat != NULL, "Checking r2lTrnMat is not null");

    mitk::PointSet::Pointer pointSet = mitk::PointSet::New();
    mitk::StereoTagExtractor::Pointer extractor = mitk::StereoTagExtractor::New();
    extractor->ExtractPoints(leftMitkImage, rightMitkImage, 0.01, 0.125, *leftIntMat, *rightIntMat, *r2lRotMat, *r2lTrnMat, pointSet);

    MITK_TEST_CONDITION_REQUIRED(pointSet->GetSize() == 2,".. Testing we got 2 points out, and we got " << pointSet->GetSize());

    cvReleaseMat(&leftIntMat);
    cvReleaseMat(&rightIntMat);
    cvReleaseMat(&r2lRotMat);
    cvReleaseMat(&r2lTrnMat);

    MITK_TEST_OUTPUT(<< "Finished Test3DReconstruction...");
  }


  //-----------------------------------------------------------------------------
  static void TestMono(const std::string& image)
  {
    MITK_TEST_OUTPUT(<< "Starting TestMono...");

    mitk::Image::Pointer mitkImage = mitk::IOUtil::LoadImage(image);
    MITK_TEST_CONDITION_REQUIRED(mitkImage.IsNotNull(), "Checking mitkImage is not null");

    mitk::PointSet::Pointer pointSet = mitk::PointSet::New();
    mitk::MonoTagExtractor::Pointer extractor = mitk::MonoTagExtractor::New();
    extractor->ExtractPoints(mitkImage, 0.01, 0.125, pointSet);

    MITK_TEST_CONDITION_REQUIRED(pointSet->GetSize() == 5,".. Testing we got 2 points out, and we got " << pointSet->GetSize());
    MITK_TEST_OUTPUT(<< "Finished TestMono...");
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
  TagTrackingTest::TestMono(leftImage);

  MITK_TEST_END();
}

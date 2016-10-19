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
#include <mitkOpenCVFileIOUtils.h>
#include <mitkIOUtil.h>
#include <mitkCameraIntrinsics.h>
#include <niftkDistanceFromCamera.h>

int niftkDistanceFromCameraTest ( int argc, char * argv[] )
{
  // always start with this!
  MITK_TEST_BEGIN("niftkDistanceFromCameraTest");

  if (argc != 7)
  {
    std::cerr << "Usage: niftkDistanceFromCameraTest left.png right.png calib.left.intrinsic.txt calib.right.intrinsic.txt calib.r2l.txt expectedDistance" << std::endl;
    return EXIT_FAILURE;
  }

  std::string leftImageFileName = argv[1];
  std::string rightImageFileName = argv[2];
  std::string leftIntrinsicFileName = argv[3];
  std::string rightIntrinsicFileName = argv[4];
  std::string right2LeftFileName = argv[5];
  double expectedDistance = atof(argv[6]);
  double tolerance = 0.001;

  cv::Mat leftIntrinsic = cvCreateMat (3,3,CV_64FC1);
  cv::Mat leftDistortion = cvCreateMat (1,4,CV_64FC1);
  cv::Mat rightIntrinsic = cvCreateMat (3,3,CV_64FC1);
  cv::Mat rightDistortion = cvCreateMat (1,4,CV_64FC1);
  cv::Mat rightToLeftRotationMatrix = cvCreateMat (3,3,CV_64FC1);
  cv::Mat rightToLeftTranslationVector = cvCreateMat (1,3,CV_64FC1);

  mitk::Image::Pointer leftImage = mitk::IOUtil::LoadImage(leftImageFileName);
  mitk::Image::Pointer rightImage = mitk::IOUtil::LoadImage(rightImageFileName);

  mitk::LoadCameraIntrinsicsFromPlainText(leftIntrinsicFileName, &leftIntrinsic, &leftDistortion);
  mitk::LoadCameraIntrinsicsFromPlainText(rightIntrinsicFileName, &rightIntrinsic, &rightDistortion);
  mitk::LoadStereoTransformsFromPlainText(right2LeftFileName, &rightToLeftRotationMatrix, &rightToLeftTranslationVector);

  mitk::CameraIntrinsics::Pointer leftIntr = mitk::CameraIntrinsics::New();
  leftIntr->SetFocalLength(leftIntrinsic.at<double>(0, 0), leftIntrinsic.at<double>(1, 1));
  leftIntr->SetPrincipalPoint(leftIntrinsic.at<double>(0, 2), leftIntrinsic.at<double>(1, 2));

  mitk::CameraIntrinsics::Pointer rightIntr = mitk::CameraIntrinsics::New();
  rightIntr->SetFocalLength(rightIntrinsic.at<double>(0, 0), rightIntrinsic.at<double>(1, 1));
  rightIntr->SetPrincipalPoint(rightIntrinsic.at<double>(0, 2), rightIntrinsic.at<double>(1, 2));

  itk::Matrix<float, 4, 4> stereoExtr;
  stereoExtr.SetIdentity();
  for (int r = 0; r < 3; r++)
  {
    for (int c = 0; c < 3; c++)
    {
      stereoExtr[r][c] = rightToLeftRotationMatrix.at<double>(r,c);
    }
    stereoExtr[r][3] = rightToLeftTranslationVector.at<double>(0, r);
  }
  niftk::DistanceFromCamera::Pointer measurer = niftk::DistanceFromCamera::New();
  double actualDistance = measurer->GetDistance(leftImage, rightImage, leftIntr, rightIntr, stereoExtr);

  MITK_TEST_CONDITION (fabs(expectedDistance - actualDistance) < tolerance, "... expected=" << expectedDistance << ", actual=" << actualDistance << ", tol=" << tolerance);
  MITK_TEST_END();
}

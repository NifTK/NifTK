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
#include <fstream>
#include <sstream>
#include <string>
#include <CameraCalibration/mitkCameraCalibrationFacade.h>
#include <mitkOpenCVMaths.h>
#include <mitkOpenCVFileIOUtils.h>

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

void TriangulatePointPairUsingGeometryTest()
{
  //test with float matrices
  cv::Mat leftIntrinsicFloat (3,3,CV_32FC1);
  cv::Mat rightIntrinsicFloat (3,3,CV_32FC1);
  cv::Mat rightToLeftRotationMatrixFloat (3,3,CV_32FC1);
  cv::Mat rightToLeftTranslationVectorFloat (1,3,CV_32FC1);
  
  cv::Mat leftIntrinsicDouble (3,3,CV_64FC1);
  cv::Mat rightIntrinsicDouble (3,3,CV_64FC1);
  cv::Mat rightToLeftRotationMatrixDouble (3,3,CV_64FC1);
  cv::Mat rightToLeftTranslationVectorDouble (1,3,CV_64FC1);

  // a point slightly offset and 100 mm infront of the left lens
  cv::Point3d inputPoint (10,10,100);

  //some real calibration results
  leftIntrinsicFloat.at<float>(0,0) = 2133.494534;
  leftIntrinsicFloat.at<float>(0,1) = 0.0;
  leftIntrinsicFloat.at<float>(0,2) = 918.8730165;
  leftIntrinsicFloat.at<float>(1,0) = 0.0;
  leftIntrinsicFloat.at<float>(1,1) = 1078.524701;
  leftIntrinsicFloat.at<float>(1,2) = 252.3014626;
  leftIntrinsicFloat.at<float>(2,0) = 0.0;
  leftIntrinsicFloat.at<float>(2,1) = 0.0;
  leftIntrinsicFloat.at<float>(2,2) = 1.0;

  leftIntrinsicDouble.at<double>(0,0) = 2133.494534;
  leftIntrinsicDouble.at<double>(0,1) = 0.0;
  leftIntrinsicDouble.at<double>(0,2) = 918.8730165;
  leftIntrinsicDouble.at<double>(1,0) = 0.0;
  leftIntrinsicDouble.at<double>(1,1) = 1078.524701;
  leftIntrinsicDouble.at<double>(1,2) = 252.3014626;
  leftIntrinsicDouble.at<double>(2,0) = 0.0;
  leftIntrinsicDouble.at<double>(2,1) = 0.0;
  leftIntrinsicDouble.at<double>(2,2) = 1.0;

  rightIntrinsicFloat.at<float>(0,0) = 2163.274848;
  rightIntrinsicFloat.at<float>(0,1) = 0.0;
  rightIntrinsicFloat.at<float>(0,2) = 1076.481499;
  rightIntrinsicFloat.at<float>(1,0) = 0.0;
  rightIntrinsicFloat.at<float>(1,1) = 1089.176842;
  rightIntrinsicFloat.at<float>(1,2) = 240.2273278;
  rightIntrinsicFloat.at<float>(2,0) = 0.0;
  rightIntrinsicFloat.at<float>(2,1) = 0.0;
  rightIntrinsicFloat.at<float>(2,2) = 1.0;

  rightIntrinsicDouble.at<double>(0,0) = 2163.274848;
  rightIntrinsicDouble.at<double>(0,1) = 0.0;
  rightIntrinsicDouble.at<double>(0,2) = 1076.481499;
  rightIntrinsicDouble.at<double>(1,0) = 0.0;
  rightIntrinsicDouble.at<double>(1,1) = 1089.176842;
  rightIntrinsicDouble.at<double>(1,2) = 240.2273278;
  rightIntrinsicDouble.at<double>(2,0) = 0.0;
  rightIntrinsicDouble.at<double>(2,1) = 0.0;
  rightIntrinsicDouble.at<double>(2,2) = 1.0;

  rightToLeftRotationMatrixFloat.at<float>(0,0) = 0.9999999017;
  rightToLeftRotationMatrixFloat.at<float>(0,1) = -0.0001167696428;
  rightToLeftRotationMatrixFloat.at<float>(0,2) = 0.0004277580889;
  rightToLeftRotationMatrixFloat.at<float>(1,0) = 0.0001263429124;
  rightToLeftRotationMatrixFloat.at<float>(1,1) = 0.9997479844 ;
  rightToLeftRotationMatrixFloat.at<float>(1,2) = -0.02244886822;
  rightToLeftRotationMatrixFloat.at<float>(2,0) = -0.0004250289408;
  rightToLeftRotationMatrixFloat.at<float>(2,1) =  0.02244892005;
  rightToLeftRotationMatrixFloat.at<float>(2,2) = 0.9997479009;

  rightToLeftRotationMatrixDouble.at<double>(0,0) = 0.9999999017;
  rightToLeftRotationMatrixDouble.at<double>(0,1) = -0.0001167696428;
  rightToLeftRotationMatrixDouble.at<double>(0,2) = 0.0004277580889;
  rightToLeftRotationMatrixDouble.at<double>(1,0) = 0.0001263429124;
  rightToLeftRotationMatrixDouble.at<double>(1,1) = 0.9997479844 ;
  rightToLeftRotationMatrixDouble.at<double>(1,2) = -0.02244886822;
  rightToLeftRotationMatrixDouble.at<double>(2,0) = -0.0004250289408;
  rightToLeftRotationMatrixDouble.at<double>(2,1) =  0.02244892005;
  rightToLeftRotationMatrixDouble.at<double>(2,2) = 0.9997479009;

  rightToLeftTranslationVectorFloat.at<float>(0,0) = 4.887636772; 
  rightToLeftTranslationVectorFloat.at<float>(0,1) = 0.3970725601;
  rightToLeftTranslationVectorFloat.at<float>(0,2) = 0.3233443251;
  
  rightToLeftTranslationVectorDouble.at<double>(0,0) = 4.887636772; 
  rightToLeftTranslationVectorDouble.at<double>(0,1) = 0.3970725601;
  rightToLeftTranslationVectorDouble.at<double>(0,2) = 0.3233443251;

  //and this is what we should project to
  cv::Point2d leftScreenPoint ( 1132.2225, 360.1539 );
  cv::Point2d rightScreenPoint (1186.8113 , 369.9688);
//  cv::Point2d rightScreenPoint (1397.7766,328.5475);
  std::pair <cv::Point2d, cv::Point2d> projectedPoints (leftScreenPoint, rightScreenPoint);

  cv::Point3d triangulatedFloat = mitk::TriangulatePointPairUsingGeometry (
      projectedPoints, 
      leftIntrinsicFloat, rightIntrinsicFloat, rightToLeftRotationMatrixFloat, rightToLeftTranslationVectorFloat );
  
  cv::Point3d triangulatedDouble = mitk::TriangulatePointPairUsingGeometry (
      projectedPoints, 
      leftIntrinsicDouble, rightIntrinsicDouble, rightToLeftRotationMatrixDouble, rightToLeftTranslationVectorDouble );

  
  MITK_TEST_CONDITION(mitk::NearlyEqual (triangulatedFloat, inputPoint, 0.025) , "Check triangulation for floating point parameters " << triangulatedFloat);
  MITK_TEST_CONDITION(mitk::NearlyEqual (triangulatedDouble, inputPoint, 0.025) , "Check triangulation for floating point parameters " << triangulatedDouble);
  
}

void UndistortTest()
{
  //test functioning of mitk::UndisrtortPoints with 4 and 5 length distortion vectors and float and double intrinsisc
  cv::Mat intrinsicFloat (3,3,CV_32FC1);
  cv::Mat distortionFourFloat (1,4,CV_32FC1);
  cv::Mat distortionFiveFloat (1,5,CV_32FC1);
  
  cv::Mat intrinsicDouble (3,3,CV_64FC1);
  cv::Mat distortionFourDouble (1,4,CV_64FC1);
  cv::Mat distortionFiveDouble (1,5,CV_64FC1);

  //some real calibration results
  intrinsicFloat.at<float>(0,0) = 2133.494534;
  intrinsicFloat.at<float>(0,1) = 0.0;
  intrinsicFloat.at<float>(0,2) = 918.8730165;
  intrinsicFloat.at<float>(1,0) = 0.0;
  intrinsicFloat.at<float>(1,1) = 1078.524701;
  intrinsicFloat.at<float>(1,2) = 252.3014626;
  intrinsicFloat.at<float>(2,0) = 0.0;
  intrinsicFloat.at<float>(2,1) = 0.0;
  intrinsicFloat.at<float>(2,2) = 1.0;

  intrinsicDouble.at<double>(0,0) = 2133.494534;
  intrinsicDouble.at<double>(0,1) = 0.0;
  intrinsicDouble.at<double>(0,2) = 918.8730165;
  intrinsicDouble.at<double>(1,0) = 0.0;
  intrinsicDouble.at<double>(1,1) = 1078.524701;
  intrinsicDouble.at<double>(1,2) = 252.3014626;
  intrinsicDouble.at<double>(2,0) = 0.0;
  intrinsicDouble.at<double>(2,1) = 0.0;
  intrinsicDouble.at<double>(2,2) = 1.0;

  distortionFourFloat.at<float>(0,0) = -0.287929;
  distortionFourFloat.at<float>(0,1) = 0.48469;
  distortionFourFloat.at<float>(0,2) = -0.00593087;
  distortionFourFloat.at<float>(0,3) = 0.00757244;

  distortionFiveFloat.at<float>(0,0) = -0.287929;
  distortionFiveFloat.at<float>(0,1) = 0.48469;
  distortionFiveFloat.at<float>(0,2) = -0.00593087;
  distortionFiveFloat.at<float>(0,3) = 0.00757244;
  distortionFiveFloat.at<float>(0,4) = 0.0;

  distortionFourDouble.at<double>(0,0) = -0.287929;
  distortionFourDouble.at<double>(0,1) = 0.48469;
  distortionFourDouble.at<double>(0,2) = -0.00593087;
  distortionFourDouble.at<double>(0,3) = 0.00757244;

  distortionFiveDouble.at<double>(0,0) = -0.287929;
  distortionFiveDouble.at<double>(0,1) = 0.48469;
  distortionFiveDouble.at<double>(0,2) = -0.00593087;
  distortionFiveDouble.at<double>(0,3) = 0.00757244;
  distortionFiveDouble.at<double>(0,4) = 0.0;

  std::vector <cv::Point2d> inputPoints;
  std::vector <cv::Point2d> outPointsFourFloat;
  std::vector <cv::Point2d> outPointsFiveFloat;
  std::vector <cv::Point2d> outPointsFourDouble;
  std::vector <cv::Point2d> outPointsFiveDouble;

  for ( double x = 5 ; x < 1920 ;  x += 150 ) 
  {
    for ( double y = 5 ; y < 1080 ; y += 150 )
    {
      inputPoints.push_back ( cv::Point2d ( x,y ) );
    }
  }

  mitk::UndistortPoints ( inputPoints, intrinsicFloat, distortionFourFloat, outPointsFourFloat );
  mitk::UndistortPoints ( inputPoints, intrinsicFloat, distortionFiveFloat, outPointsFiveFloat );
  mitk::UndistortPoints ( inputPoints, intrinsicDouble, distortionFourDouble, outPointsFourDouble );
  mitk::UndistortPoints ( inputPoints, intrinsicDouble, distortionFiveDouble, outPointsFiveDouble );

  for ( unsigned int i = 0 ; i < inputPoints.size() ; i++ ) 
  {
    MITK_TEST_CONDITION (mitk::NearlyEqual (outPointsFourFloat[i], outPointsFiveFloat[i], 0.001), "Points equal " << i << outPointsFourFloat[i] <<outPointsFiveFloat[i]);;
    MITK_TEST_CONDITION (mitk::NearlyEqual (outPointsFourFloat[i], outPointsFourDouble[i], 0.001), "Points equal " << i << outPointsFourFloat[i] <<outPointsFiveFloat[i]);;
    MITK_TEST_CONDITION (mitk::NearlyEqual (outPointsFourFloat[i], outPointsFiveDouble[i], 0.001), "Points equal " << i << outPointsFourFloat[i] <<outPointsFiveFloat[i]);;
  }
  
}

void GetRayTest ()
{

  cv::Mat intrinsicFloat (3,3,CV_32FC1);
  cv::Mat intrinsicDouble (3,3,CV_64FC1);

  //some real calibration results
  intrinsicFloat.at<float>(0,0) = 2000.0;
  intrinsicFloat.at<float>(0,1) = 0.0;
  intrinsicFloat.at<float>(0,2) = 900.0;
  intrinsicFloat.at<float>(1,0) = 0.0;
  intrinsicFloat.at<float>(1,1) = 1000.0;
  intrinsicFloat.at<float>(1,2) = 250.0;
  intrinsicFloat.at<float>(2,0) = 0.0;
  intrinsicFloat.at<float>(2,1) = 0.0;
  intrinsicFloat.at<float>(2,2) = 1.0;

  intrinsicDouble.at<double>(0,0) = 2000.0;
  intrinsicDouble.at<double>(0,1) = 0.0;
  intrinsicDouble.at<double>(0,2) = 900.0;
  intrinsicDouble.at<double>(1,0) = 0.0;
  intrinsicDouble.at<double>(1,1) = 1000.0;
  intrinsicDouble.at<double>(1,2) = 250.0;
  intrinsicDouble.at<double>(2,0) = 0.0;
  intrinsicDouble.at<double>(2,1) = 0.0;
  intrinsicDouble.at<double>(2,2) = 1.0;

  cv::Point2d screenDouble ( 100, 200 );
  cv::Point2f screenFloat ( 100, 200 );

  std::pair < cv::Point3d, cv::Point3d > rayFloat = mitk::GetRay ( screenFloat, intrinsicFloat , 1.0);
  std::pair < cv::Point3d, cv::Point3d > rayDouble = mitk::GetRay ( screenDouble, intrinsicDouble, 1.0 );

  MITK_TEST_CONDITION (mitk::NearlyEqual (rayFloat.first, rayDouble.first, 1e-6), "Get ray float and double equal ");
  MITK_TEST_CONDITION (mitk::NearlyEqual (rayFloat.second, rayDouble.second, 1e-6), "Get ray float and double equal ");
  MITK_TEST_CONDITION (mitk::NearlyEqual (rayDouble.first, cv::Point3d ( 0.0, 0.0, 0.0) , 1e-6), "Get ray origin is zero");
  MITK_TEST_CONDITION (mitk::NearlyEqual (rayDouble.second, cv::Point3d ( -0.4, -0.05, 1.0) , 1e-6), "Get ray second point is -0.4, -0.05, 1.0" << rayDouble.second);

}

//-----------------------------------------------------------------------------
int mitkCameraCalibrationFacadeTest(int argc, char * argv[])
{
  // always start with this!
  MITK_TEST_BEGIN("mitkCameraCalibrationFacadeTest");

  CheckExceptionsForLoadingFromPlaintext();
  TriangulatePointPairUsingGeometryTest();
  UndistortTest();
  GetRayTest();

  MITK_TEST_END();
}

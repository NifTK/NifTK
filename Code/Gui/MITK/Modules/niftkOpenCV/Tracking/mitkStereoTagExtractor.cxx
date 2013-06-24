/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkStereoTagExtractor.h"
#include "mitkTagTrackingFacade.h"
#include <mitkImageToOpenCVImageFilter.h>
#include <mitkCameraIntrinsicsProperty.h>
#include <cv.h>
#include <Undistortion.h>

namespace mitk {

//-----------------------------------------------------------------------------
StereoTagExtractor::StereoTagExtractor()
{

}


//-----------------------------------------------------------------------------
StereoTagExtractor::~StereoTagExtractor()
{

}


//-----------------------------------------------------------------------------
void StereoTagExtractor::ExtractPoints(const mitk::Image::Pointer leftImage,
                   const mitk::Image::Pointer rightImage,
                   const float& minSize,
                   const float& maxSize,
                   mitk::PointSet::Pointer pointSet,
                   const vtkMatrix4x4* cameraToWorld
                   )
{
  CvMat* leftCameraIntrinsics  = cvCreateMat(3, 3, CV_32FC1);
  CvMat* rightCameraIntrinsics  = cvCreateMat(3, 3, CV_32FC1);
  CvMat* rightToLeftRotationMatrix  = cvCreateMat(3, 3, CV_32FC1);
  CvMat* rightToLeftRotationVector  = cvCreateMat(1, 3, CV_32FC1);
  CvMat* rightToLeftTranslationVector  = cvCreateMat(1, 3, CV_32FC1);

  itk::Matrix<float, 4, 4> txf;
  txf.SetIdentity();

  niftk::Undistortion::MatrixProperty::Pointer leftToRightMatrixProp = niftk::Undistortion::MatrixProperty::New(txf);
  mitk::CameraIntrinsicsProperty::Pointer leftIntrinsicsProp = mitk::CameraIntrinsicsProperty::New();
  mitk::CameraIntrinsicsProperty::Pointer rightIntrinsicsProp = mitk::CameraIntrinsicsProperty::New();

  leftIntrinsicsProp = dynamic_cast<mitk::CameraIntrinsicsProperty*>(leftImage->GetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName).GetPointer());
  rightIntrinsicsProp = dynamic_cast<mitk::CameraIntrinsicsProperty*>(rightImage->GetProperty(niftk::Undistortion::s_CameraCalibrationPropertyName).GetPointer());
  leftToRightMatrixProp = dynamic_cast<niftk::Undistortion::MatrixProperty*>(rightImage->GetProperty(niftk::Undistortion::s_StereoRigTransformationPropertyName).GetPointer());

  mitk::CameraIntrinsics::Pointer leftCam = mitk::CameraIntrinsics::New();
  mitk::CameraIntrinsics::Pointer rightCam = mitk::CameraIntrinsics::New();

  if (leftIntrinsicsProp.IsNotNull()
    && rightIntrinsicsProp.IsNotNull()
    && leftToRightMatrixProp.IsNotNull()
  )
  {
    leftCam = leftIntrinsicsProp->GetValue();
    rightCam = rightIntrinsicsProp->GetValue();
    txf = leftToRightMatrixProp->GetValue();
  }
  else
  {
    // invent some stuff based on image dimensions
    // ToDo: Undistortion has a bit like this.
    unsigned int w = leftImage->GetDimension(0);
    unsigned int h = leftImage->GetDimension(1);

    mitk::Point3D::ValueType  focal[3] = {(float) std::max(w, h), (float) std::max(w, h), 1};
    mitk::Point3D::ValueType  princ[3] = {(float) w / 2, (float) h / 2, 1};
    mitk::Point4D::ValueType  disto[4] = {0, 0, 0, 0};

    leftCam->SetIntrinsics(mitk::Point3D(focal), mitk::Point3D(princ), mitk::Point4D(disto));
    rightCam->SetIntrinsics(mitk::Point3D(focal), mitk::Point3D(princ), mitk::Point4D(disto));
    txf[0][3] = 10;
  }

  CV_MAT_ELEM(*leftCameraIntrinsics, float, 0, 0) = leftCam->GetFocalLengthX();
  CV_MAT_ELEM(*leftCameraIntrinsics, float, 0, 1) = 0;
  CV_MAT_ELEM(*leftCameraIntrinsics, float, 0, 2) = leftCam->GetPrincipalPointX();
  CV_MAT_ELEM(*leftCameraIntrinsics, float, 1, 0) = 0;
  CV_MAT_ELEM(*leftCameraIntrinsics, float, 1, 1) = leftCam->GetFocalLengthY();
  CV_MAT_ELEM(*leftCameraIntrinsics, float, 1, 2) = leftCam->GetPrincipalPointY();
  CV_MAT_ELEM(*leftCameraIntrinsics, float, 2, 0) = 0;
  CV_MAT_ELEM(*leftCameraIntrinsics, float, 2, 1) = 0;
  CV_MAT_ELEM(*leftCameraIntrinsics, float, 2, 2) = 1;

  CV_MAT_ELEM(*rightCameraIntrinsics, float, 0, 0) = rightCam->GetFocalLengthX();
  CV_MAT_ELEM(*rightCameraIntrinsics, float, 0, 1) = 0;
  CV_MAT_ELEM(*rightCameraIntrinsics, float, 0, 2) = rightCam->GetPrincipalPointX();
  CV_MAT_ELEM(*rightCameraIntrinsics, float, 1, 0) = 0;
  CV_MAT_ELEM(*rightCameraIntrinsics, float, 1, 1) = rightCam->GetFocalLengthY();
  CV_MAT_ELEM(*rightCameraIntrinsics, float, 1, 2) = rightCam->GetPrincipalPointY();
  CV_MAT_ELEM(*rightCameraIntrinsics, float, 2, 0) = 0;
  CV_MAT_ELEM(*rightCameraIntrinsics, float, 2, 1) = 0;
  CV_MAT_ELEM(*rightCameraIntrinsics, float, 2, 2) = 1;

  for (int i = 0; i < 3; i++)
  {
    for (int j  = 0; j < 3; j++)
    {
      CV_MAT_ELEM(*rightToLeftRotationMatrix, float, i, j) = txf[i][j];
    }
    CV_MAT_ELEM(*rightToLeftTranslationVector, float, 0, i) = txf[i][3];
  }
  cvRodrigues2(rightToLeftRotationMatrix, rightToLeftRotationVector);

  this->ExtractPoints(leftImage, rightImage,
                      minSize, maxSize,
                      *leftCameraIntrinsics, *rightCameraIntrinsics,
                      *rightToLeftRotationVector, *rightToLeftTranslationVector,
                      pointSet, cameraToWorld);

  cvReleaseMat(&leftCameraIntrinsics);
  cvReleaseMat(&rightCameraIntrinsics);
  cvReleaseMat(&rightToLeftRotationVector);
  cvReleaseMat(&rightToLeftRotationMatrix);
  cvReleaseMat(&rightToLeftTranslationVector);
}


//-----------------------------------------------------------------------------
void StereoTagExtractor::ExtractPoints(const mitk::Image::Pointer leftImage,
                                       const mitk::Image::Pointer rightImage,
                                       const float& minSize,
                                       const float& maxSize,
                                       const CvMat& leftCameraIntrinsics,
                                       const CvMat& rightCameraIntrinsics,
                                       const CvMat& rightToLeftRotationVector,
                                       const CvMat& rightToLeftTranslationVector,
                                       mitk::PointSet::Pointer pointSet,
                                       const vtkMatrix4x4* cameraToWorld
                                      )
{
  pointSet->Clear();

  mitk::ImageToOpenCVImageFilter::Pointer leftOpenCVFilter = mitk::ImageToOpenCVImageFilter::New();
  leftOpenCVFilter->SetImage(leftImage);

  IplImage *leftIm = leftOpenCVFilter->GetOpenCVImage();
  cv::Mat left(leftIm);

  mitk::ImageToOpenCVImageFilter::Pointer rightOpenCVFilter = mitk::ImageToOpenCVImageFilter::New();
  rightOpenCVFilter->SetImage(rightImage);

  IplImage *rightIm = rightOpenCVFilter->GetOpenCVImage();
  cv::Mat right(rightIm);

  cv::Mat leftInt(&leftCameraIntrinsics);
  cv::Mat rightInt(&rightCameraIntrinsics);
  cv::Mat r2lRot(&rightToLeftRotationVector);
  cv::Mat r2lTran(&rightToLeftTranslationVector);

  std::map<int, cv::Point3f> result = mitk::DetectMarkerPairs(
    left,
    right,
    leftInt,
    rightInt,
    r2lRot,
    r2lTran,
    minSize,
    maxSize
    );

  cv::Point3f extractedPoint;
  mitk::PointSet::PointType outputPoint;

  std::map<int, cv::Point3f>::iterator iter;
  for (iter = result.begin(); iter != result.end(); ++iter)
  {
    extractedPoint = (*iter).second;
    outputPoint[0] = extractedPoint.x;
    outputPoint[1] = extractedPoint.y;
    outputPoint[2] = extractedPoint.z;
    TransformPointsByCameraToWorld(const_cast<vtkMatrix4x4*>(cameraToWorld), outputPoint);
    pointSet->InsertPoint((*iter).first, outputPoint);
  }

  cvReleaseImage(&leftIm);
  cvReleaseImage(&rightIm);
}


//-----------------------------------------------------------------------------
} // end namespace

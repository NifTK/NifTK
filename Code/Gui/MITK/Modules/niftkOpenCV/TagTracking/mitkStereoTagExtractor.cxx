/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkStereoTagExtractor.h"
#include <mitkCameraCalibrationFacade.h>
#include <mitkCameraIntrinsicsProperty.h>
#include <mitkImageReadAccessor.h>
#include <mitkImageWriteAccessor.h>
#include <mitkPointUtils.h>
#include <mitkMIDASImageUtils.h>
#include <aruco/aruco.h>
#include <Undistortion.h>

namespace mitk {

/**
 * \brief PIMPL pattern, private implementation to reduce visibility
 * of dependencies on OpenCV and ARUCO.
 */
class StereoTagExtractorPrivate {

public:

  StereoTagExtractorPrivate();
  ~StereoTagExtractorPrivate();

  void ExtractPoints(const mitk::Image::Pointer leftImage,
                     const mitk::Image::Pointer rightImage,
                     const float& minSize,
                     const float& maxSize,
                     const int& blockSize,
                     const int& offset,
                     const CvMat& leftCameraIntrinsics,
                     const CvMat& rightCameraIntrinsics,
                     const CvMat& rightToLeftRotationVector,
                     const CvMat& rightToLeftTranslationVector,
                     const vtkMatrix4x4* cameraToWorld,
                     mitk::PointSet::Pointer pointSet,
                     mitk::PointSet::Pointer surfaceNormals
                     );

  /**
   * \brief Overloaded interface for other method, extracting the necessary matrices off of the mitk::Image
   */
  void ExtractPoints(const mitk::Image::Pointer leftImage,
                     const mitk::Image::Pointer rightImage,
                     const float& minSize,
                     const float& maxSize,
                     const int& blockSize,
                     const int& offset,
                     const vtkMatrix4x4* cameraToWorld,
                     mitk::PointSet::Pointer pointSet,
                     mitk::PointSet::Pointer surfaceNormals
                     );
private:

  // So, these are persistant between calls to the MonoTagExtractor for performance reasons.
  aruco::MarkerDetector   m_LeftDetector;
  aruco::CameraParameters m_LeftCameraParams;
  cv::Mat                 m_LeftGreyImage;
  cv::Mat                 m_LeftResizedImage;
  aruco::MarkerDetector   m_RightDetector;
  aruco::CameraParameters m_RightCameraParams;
  cv::Mat                 m_RightGreyImage;
  cv::Mat                 m_RightResizedImage;
};


//-----------------------------------------------------------------------------
StereoTagExtractorPrivate::StereoTagExtractorPrivate()
{

}


//-----------------------------------------------------------------------------
StereoTagExtractorPrivate::~StereoTagExtractorPrivate()
{

}


//-----------------------------------------------------------------------------
void StereoTagExtractorPrivate::ExtractPoints(const mitk::Image::Pointer leftImage,
                   const mitk::Image::Pointer rightImage,
                   const float& minSize,
                   const float& maxSize,
                   const int& blockSize,
                   const int& offset,
                   const vtkMatrix4x4* cameraToWorld,
                   mitk::PointSet::Pointer pointSet,
                   mitk::PointSet::Pointer surfaceNormals
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
                      blockSize, offset,
                      *leftCameraIntrinsics, *rightCameraIntrinsics,
                      *rightToLeftRotationVector, *rightToLeftTranslationVector, cameraToWorld,
                      pointSet, surfaceNormals);

  cvReleaseMat(&leftCameraIntrinsics);
  cvReleaseMat(&rightCameraIntrinsics);
  cvReleaseMat(&rightToLeftRotationVector);
  cvReleaseMat(&rightToLeftRotationMatrix);
  cvReleaseMat(&rightToLeftTranslationVector);
}


//-----------------------------------------------------------------------------
void StereoTagExtractorPrivate::ExtractPoints(const mitk::Image::Pointer leftImage,
                                       const mitk::Image::Pointer rightImage,
                                       const float& minSize,
                                       const float& maxSize,
                                       const int& blockSize,
                                       const int& offset,
                                       const CvMat& leftCameraIntrinsics,
                                       const CvMat& rightCameraIntrinsics,
                                       const CvMat& rightToLeftRotationVector,
                                       const CvMat& rightToLeftTranslationVector,
                                       const vtkMatrix4x4* cameraToWorld,
                                       mitk::PointSet::Pointer pointSet,
                                       mitk::PointSet::Pointer surfaceNormals
                                      )
{
  // For each iteration the point set is erased.
  pointSet->Clear();

  // The user might pass in NULL surface normals, so only clear if necessary.
  if (surfaceNormals.IsNotNull())
  {
    surfaceNormals->Clear();
  }
  
  // Retrieve scaling parameters, as input 'image' may have anisotropic voxels.
  mitk::Vector3D aspect = mitk::GetXYAspectRatio(leftImage);

  // This creates a cv::Mat by wrapping and not copying the data in 'leftImage'.
  mitk::ImageWriteAccessor  leftAccess(leftImage);
  void* leftPointer = leftAccess.GetData();
  IplImage  leftIpl;
  cvInitImageHeader(&leftIpl, cvSize((int) leftImage->GetDimension(0), (int) leftImage->GetDimension(1)), leftImage->GetPixelType().GetBitsPerComponent(), leftImage->GetPixelType().GetNumberOfComponents());
  cvSetData(&leftIpl, leftPointer, leftImage->GetDimension(0) * (leftImage->GetPixelType().GetBitsPerComponent() / 8) * leftImage->GetPixelType().GetNumberOfComponents());
  cv::Mat leftColour(&leftIpl);

  // This creates a cv::Mat by wrapping and not copying the data in 'rightImage'.
  mitk::ImageWriteAccessor  rightAccess(rightImage);
  void* rightPointer = rightAccess.GetData();
  IplImage  rightIpl;
  cvInitImageHeader(&rightIpl, cvSize((int) rightImage->GetDimension(0), (int) rightImage->GetDimension(1)), rightImage->GetPixelType().GetBitsPerComponent(), rightImage->GetPixelType().GetNumberOfComponents());
  cvSetData(&rightIpl, rightPointer, rightImage->GetDimension(0) * (rightImage->GetPixelType().GetBitsPerComponent() / 8) * rightImage->GetPixelType().GetNumberOfComponents());
  cv::Mat rightColour(&rightIpl);

  // This converts our RGBA images to grey.
  cv::cvtColor(leftColour, m_LeftGreyImage, CV_RGBA2GRAY);
  cv::cvtColor(rightColour, m_RightGreyImage, CV_RGBA2GRAY);

  // This will rescale it to the right size, as we know our video capture can produce input images with anisotropic voxels.
  cv::resize(m_LeftGreyImage, m_LeftResizedImage, cv::Size(0, 0), aspect[0], aspect[1], cv::INTER_NEAREST);
  cv::resize(m_RightGreyImage, m_RightResizedImage, cv::Size(0, 0), aspect[0], aspect[1], cv::INTER_NEAREST);

  // Wrap the C-style matrices to C++ style.
  cv::Mat leftInt(&leftCameraIntrinsics);
  cv::Mat rightInt(&rightCameraIntrinsics);
  cv::Mat r2lRot(&rightToLeftRotationVector);
  cv::Mat r2lTran(&rightToLeftTranslationVector);

  // Eyes.... LEFT!
  std::vector<aruco::Marker> leftMarkers;
  m_LeftDetector.setMinMaxSize(minSize, maxSize);
  m_LeftDetector.setThresholdMethod(aruco::MarkerDetector::ADPT_THRES);
  m_LeftDetector.setThresholdParams(blockSize, offset);
  m_LeftDetector.detect(m_LeftResizedImage, leftMarkers, m_LeftCameraParams);

  // Eyes.... RIGHT!
  std::vector<aruco::Marker> rightMarkers;
  m_RightDetector.setMinMaxSize(minSize, maxSize);
  m_RightDetector.setThresholdMethod(aruco::MarkerDetector::ADPT_THRES);
  m_RightDetector.setThresholdParams(blockSize, offset);
  m_RightDetector.detect(m_RightResizedImage, rightMarkers, m_RightCameraParams);

  // Now we find corresponding markers
  for (unsigned int i = 0; i < leftMarkers.size(); ++i)
  {
    int pointID = leftMarkers[i].id;

    for (unsigned int j = 0; j < rightMarkers.size(); ++j)
    {
      // Check if we have valid points detected in both left and right views.
      if (rightMarkers[j].id == pointID && leftMarkers[i].isValid() && rightMarkers[j].isValid())
      {
        // Extract and triangulate corresponding points.
        // We are assuming that each marker has similarly ordered point corners.
        // We rescale image coordinates back, as camera calibration is assumed to be done on aspect ratio of 1:1.

        cv::Point2f leftMarker;
        cv::Point2f rightMarker;
        std::vector<std::pair<cv::Point2f, cv::Point2f> > pairs;

        // If we are doing normals, we need the 4 corners.
        if (surfaceNormals.IsNotNull())
        {
          for (int k = 0; k < 4; k++)
          {
            leftMarker = leftMarkers[i][k];
            leftMarker.x /= aspect[0];
            leftMarker.y /= aspect[1];

            rightMarker = rightMarkers[j][k];
            rightMarker.x /= aspect[0];
            rightMarker.y /= aspect[1];

            std::pair<cv::Point2f, cv::Point2f> pair(leftMarker, rightMarker);
            pairs.push_back(pair);
          }
        }

        // We always need the centre point, calculated by ARUCO.
        leftMarker = leftMarkers[i].getCenter();
        leftMarker.x /= aspect[0];
        leftMarker.y /= aspect[1];

        rightMarker = rightMarkers[j].getCenter();
        rightMarker.x /= aspect[0];
        rightMarker.y /= aspect[1];

        std::pair<cv::Point2f, cv::Point2f> centrePoint(leftMarker, rightMarker);
        pairs.push_back(centrePoint);

        std::vector<cv::Point3f> pointsIn3D = mitk::TriangulatePointPairs(
            pairs,
            leftInt,
            rightInt,
            r2lRot,
            r2lTran
            );

        if (surfaceNormals.IsNotNull())
        {
          assert(pointsIn3D.size() == 5);

          mitk::Point3D a, b, c, outputNormal;
          a[0] =  pointsIn3D[0].x;
          a[1] = -pointsIn3D[0].y;
          a[2] = -pointsIn3D[0].z;
          b[0] =  pointsIn3D[1].x;
          b[1] = -pointsIn3D[1].y;
          b[2] = -pointsIn3D[1].z;
          c[0] =  pointsIn3D[2].x;
          c[1] = -pointsIn3D[2].y;
          c[2] = -pointsIn3D[2].z;
          mitk::ComputeNormalFromPoints(a, b, c, outputNormal);

          mitk::PointSet::PointType outputPoint;

          outputPoint[0] =  pointsIn3D[4].x;
          outputPoint[1] = -pointsIn3D[4].y;
          outputPoint[2] = -pointsIn3D[4].z;

          TransformPointByVtkMatrix(const_cast<vtkMatrix4x4*>(cameraToWorld), false, outputPoint);
          TransformPointByVtkMatrix(const_cast<vtkMatrix4x4*>(cameraToWorld), true, outputNormal);

          pointSet->InsertPoint(pointID, outputPoint);
          surfaceNormals->InsertPoint(pointID, outputNormal);

        }
        else
        {
          assert(pointsIn3D.size() == 1);

          mitk::PointSet::PointType outputPoint;
          outputPoint[0] =  pointsIn3D[0].x;
          outputPoint[1] = -pointsIn3D[0].y;
          outputPoint[2] = -pointsIn3D[0].z;

          TransformPointByVtkMatrix(const_cast<vtkMatrix4x4*>(cameraToWorld), false, outputPoint);
          pointSet->InsertPoint(pointID, outputPoint);

        }

        pointSet->UpdateOutputInformation();

      } // end if valid point
    } // end for each right marker
  } // end for each left marker
}


//-----------------------------------------------------------------------------
StereoTagExtractor::StereoTagExtractor()
: m_PIMPL(new StereoTagExtractorPrivate())
{
}


//-----------------------------------------------------------------------------
StereoTagExtractor::~StereoTagExtractor()
{
  if (m_PIMPL != NULL)
  {
    delete m_PIMPL;
  }
}


//-----------------------------------------------------------------------------
void StereoTagExtractor::ExtractPoints(const mitk::Image::Pointer leftImage,
                   const mitk::Image::Pointer rightImage,
                   const float& minSize,
                   const float& maxSize,
                   const int& blockSize,
                   const int& offset,
                   const vtkMatrix4x4* cameraToWorld,
                   mitk::PointSet::Pointer pointSet,
                   mitk::PointSet::Pointer surfaceNormals
                   )
{
  m_PIMPL->ExtractPoints(leftImage, rightImage, minSize, maxSize, blockSize, offset, cameraToWorld, pointSet, surfaceNormals);
}


//-----------------------------------------------------------------------------
void StereoTagExtractor::ExtractPoints(const mitk::Image::Pointer leftImage,
                                       const mitk::Image::Pointer rightImage,
                                       const float& minSize,
                                       const float& maxSize,
                                       const int& blockSize,
                                       const int& offset,
                                       const CvMat& leftCameraIntrinsics,
                                       const CvMat& rightCameraIntrinsics,
                                       const CvMat& rightToLeftRotationVector,
                                       const CvMat& rightToLeftTranslationVector,
                                       const vtkMatrix4x4* cameraToWorld,
                                       mitk::PointSet::Pointer pointSet,
                                       mitk::PointSet::Pointer surfaceNormals
                                      )
{
  m_PIMPL->ExtractPoints(leftImage, rightImage, minSize, maxSize, blockSize, offset, leftCameraIntrinsics, rightCameraIntrinsics, rightToLeftRotationVector, rightToLeftTranslationVector, cameraToWorld, pointSet, surfaceNormals);
}

//-----------------------------------------------------------------------------
} // end namespace

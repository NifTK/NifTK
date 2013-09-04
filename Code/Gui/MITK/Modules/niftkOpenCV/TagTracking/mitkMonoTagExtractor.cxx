/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMonoTagExtractor.h"
#include <mitkCommon.h>
#include <mitkImageReadAccessor.h>
#include <mitkImageWriteAccessor.h>
#include <mitkPointUtils.h>
#include <mitkMIDASImageUtils.h>
#include <cv.h>
#include <aruco/aruco.h>

namespace mitk {

/**
 * \brief PIMPL pattern, private implementation to reduce visibility
 * of dependencies on OpenCV and ARUCO.
 */
class MonoTagExtractorPrivate {

public:

  MonoTagExtractorPrivate();
  ~MonoTagExtractorPrivate();

  void ExtractPoints(const mitk::Image::Pointer image,
                     const float& minSize,
                     const float& maxSize,
                     const int& blockSize,
                     const int& offset,
                     const vtkMatrix4x4* cameraToWorld,
                     mitk::PointSet::Pointer pointSet
                     );

private:

  // So, these are persistant between calls to the MonoTagExtractor for performance reasons.
  aruco::MarkerDetector   m_Detector;
  aruco::CameraParameters m_CameraParams;
  cv::Mat                 m_GreyImage;
  cv::Mat                 m_ResizedImage;
};


//-----------------------------------------------------------------------------
MonoTagExtractorPrivate::MonoTagExtractorPrivate()
{

}


//-----------------------------------------------------------------------------
MonoTagExtractorPrivate::~MonoTagExtractorPrivate()
{

}


//-----------------------------------------------------------------------------
void MonoTagExtractorPrivate::ExtractPoints(const mitk::Image::Pointer image,
                                            const float& minSize,
                                            const float& maxSize,
                                            const int& blockSize,
                                            const int& offset,
                                            const vtkMatrix4x4* cameraToWorld,
                                            mitk::PointSet::Pointer pointSet
                                           )
{
  // For each iteration the point set is erased.
  pointSet->Clear();

  // Retrieve scaling parameters, as input 'image' may have anisotropic voxels.
  mitk::Vector3D aspect = mitk::GetXYAspectRatio(image);

  // This creates a cv::Mat by wrapping and not copying the data in 'image'.
  mitk::ImageWriteAccessor  imageAccess(image);
  void* imagePointer = imageAccess.GetData();
  IplImage  imageIpl;
  cvInitImageHeader(&imageIpl, cvSize((int) image->GetDimension(0), (int) image->GetDimension(1)), image->GetPixelType().GetBitsPerComponent(), image->GetPixelType().GetNumberOfComponents());
  cvSetData(&imageIpl, imagePointer, image->GetDimension(0) * (image->GetPixelType().GetBitsPerComponent() / 8) * image->GetPixelType().GetNumberOfComponents());
  cv::Mat leftColour(&imageIpl);

  // This converts our RGBA images to grey.
  cv::cvtColor(leftColour, m_GreyImage, CV_RGBA2GRAY);

  // This will rescale it to the right size, as we know our video capture can produce input images with anisotropic voxels.
  cv::resize(m_GreyImage, m_ResizedImage, cv::Size(0, 0), aspect[0], aspect[1], cv::INTER_NEAREST);

  // Extract markers. Note: we use default m_CameraParams.
  std::vector<aruco::Marker> markers;
  m_Detector.setMinMaxSize(minSize, maxSize);
  m_Detector.setThresholdMethod(aruco::MarkerDetector::ADPT_THRES);
  m_Detector.setThresholdParams(blockSize, offset);
  m_Detector.detect(m_ResizedImage, markers, m_CameraParams);

  // Prepare output.
  mitk::PointSet::PointType outputPoint;
  for (unsigned int i=0; i < markers.size(); i++)
  {
    cv::Point2f centrePoint = markers[i].getCenter();
    centrePoint.x /= aspect[0];
    centrePoint.y /= aspect[1];

    outputPoint[0] = centrePoint.x;
    outputPoint[1] = centrePoint.y;
    outputPoint[2] = 0;

    TransformPointByVtkMatrix(const_cast<vtkMatrix4x4*>(cameraToWorld), false, outputPoint);
    pointSet->InsertPoint(markers[i].id, outputPoint);
  }

  pointSet->Modified();
}


//-----------------------------------------------------------------------------
MonoTagExtractor::MonoTagExtractor()
: m_PIMPL(new MonoTagExtractorPrivate())
{
}


//-----------------------------------------------------------------------------
MonoTagExtractor::~MonoTagExtractor()
{
  if (m_PIMPL != NULL)
  {
    delete m_PIMPL;
  }
}


//-----------------------------------------------------------------------------
void MonoTagExtractor::ExtractPoints(const mitk::Image::Pointer image,
                                     const float& minSize,
                                     const float& maxSize,
                                     const int& blockSize,
                                     const int& offset,
                                     const vtkMatrix4x4* cameraToWorld,
                                     mitk::PointSet::Pointer pointSet
                                    )
{
  m_PIMPL->ExtractPoints(image, minSize, maxSize, blockSize, offset, cameraToWorld, pointSet);
}

//-----------------------------------------------------------------------------
} // end namespace

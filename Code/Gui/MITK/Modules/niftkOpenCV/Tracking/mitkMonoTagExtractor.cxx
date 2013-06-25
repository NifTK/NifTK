/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "mitkMonoTagExtractor.h"
#include "mitkTagTrackingFacade.h"
#include <mitkImageToOpenCVImageFilter.h>
#include <cv.h>

namespace mitk {

//-----------------------------------------------------------------------------
MonoTagExtractor::MonoTagExtractor()
{

}


//-----------------------------------------------------------------------------
MonoTagExtractor::~MonoTagExtractor()
{

}


//-----------------------------------------------------------------------------
void MonoTagExtractor::ExtractPoints(const mitk::Image::Pointer image,
                                     const float& minSize,
                                     const float& maxSize,
                                     const int& blockSize,
                                     const int& offset,
                                     mitk::PointSet::Pointer pointSet,
                                     const vtkMatrix4x4* cameraToWorld
                                    )
{
  pointSet->Clear();

  mitk::ImageToOpenCVImageFilter::Pointer filter = mitk::ImageToOpenCVImageFilter::New();
  filter->SetImage(image);

  IplImage *im = filter->GetOpenCVImage();
  cv::Mat imageWrapper(im);

  std::map<int, cv::Point2f> result = mitk::DetectMarkers(imageWrapper, minSize, maxSize);

  cv::Point2f extractedPoint;
  mitk::PointSet::PointType outputPoint;

  std::map<int, cv::Point2f>::iterator iter;
  for (iter = result.begin(); iter != result.end(); ++iter)
  {
    extractedPoint = (*iter).second;
    outputPoint[0] = extractedPoint.x;
    outputPoint[1] = extractedPoint.y;
    outputPoint[2] = 0;
    TransformPointsByCameraToWorld(const_cast<vtkMatrix4x4*>(cameraToWorld), outputPoint);
    pointSet->InsertPoint((*iter).first, outputPoint);
  }

  cvReleaseImage(&im);
}

//-----------------------------------------------------------------------------
} // end namespace

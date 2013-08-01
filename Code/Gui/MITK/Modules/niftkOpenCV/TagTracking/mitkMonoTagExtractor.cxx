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
#include <mitkImageReadAccessor.h>
#include <mitkImageWriteAccessor.h>
#include <cv.h>
#include <mitkPointUtils.h>

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
                                     const vtkMatrix4x4* cameraToWorld,
                                     mitk::PointSet::Pointer pointSet
                                    )
{
  pointSet->Clear();

  mitk::ImageWriteAccessor  imageAccess(image);
  void* imagePointer = imageAccess.GetData();

  IplImage  imageIpl;
  cvInitImageHeader(&imageIpl, cvSize((int) image->GetDimension(0), (int) image->GetDimension(1)), image->GetPixelType().GetBitsPerComponent(), image->GetPixelType().GetNumberOfComponents());
  cvSetData(&imageIpl, imagePointer, image->GetDimension(0) * (image->GetPixelType().GetBitsPerComponent() / 8) * image->GetPixelType().GetNumberOfComponents());
  cv::Mat leftColour(&imageIpl);
  cv::Mat leftGrey;
  cv::cvtColor(leftColour, leftGrey, CV_RGBA2GRAY);

  std::map<int, cv::Point2f> result = mitk::DetectMarkers(leftGrey, minSize, maxSize, blockSize, offset);

  cv::Point2f extractedPoint;
  mitk::PointSet::PointType outputPoint;

  std::map<int, cv::Point2f>::iterator iter;
  for (iter = result.begin(); iter != result.end(); ++iter)
  {
    extractedPoint = (*iter).second;
    outputPoint[0] = extractedPoint.x;
    outputPoint[1] = extractedPoint.y;
    outputPoint[2] = 0;
    TransformPointByVtkMatrix(const_cast<vtkMatrix4x4*>(cameraToWorld), false, outputPoint);
    pointSet->InsertPoint((*iter).first, outputPoint);
  }

  pointSet->Modified();
}

//-----------------------------------------------------------------------------
} // end namespace

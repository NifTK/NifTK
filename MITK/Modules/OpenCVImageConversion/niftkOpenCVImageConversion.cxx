/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkOpenCVImageConversion.h"
#include <niftkImageConversion.h>

namespace niftk
{

//-----------------------------------------------------------------------------
mitk::Image::Pointer CreateMitkImage(const IplImage* image)
{
  // FIXME: check for channel layout: rgb vs bgr
  switch (image->nChannels)
  {
    case 1:
      return CreateMitkImageInternal<unsigned char>(image->imageData,
                                                    image->nChannels,
                                                    image->width,
                                                    image->widthStep,
                                                    image->height
                                                    );
    case 3:
      return CreateMitkImageInternal<UCRGBPixelType>(image->imageData,
                                                     image->nChannels,
                                                     image->width,
                                                     image->widthStep,
                                                     image->height
                                                     );
    case 4:
      return CreateMitkImageInternal<UCRGBAPixelType>(image->imageData,
                                                      image->nChannels,
                                                      image->width,
                                                      image->widthStep,
                                                      image->height
                                                      );
  }

  assert(false);
  return 0;
}


//-----------------------------------------------------------------------------
mitk::Image::Pointer CreateMitkImage(const cv::Mat* image)
{
  IplImage* IplImg = new IplImage(*image);
  return CreateMitkImage (IplImg);
}


//-----------------------------------------------------------------------------
cv::Mat MitkImageToOpenCVMat ( const mitk::Image::Pointer image )
{
  mitk::ImageReadAccessor  inputAccess(image);
  const void* inputPointer = inputAccess.GetData();

  cv::Size size (static_cast<int>(image->GetDimension(0)), static_cast<int>(image->GetDimension(1)));
  int width = static_cast<int>(image->GetDimension(0));
  int height = static_cast<int>(image->GetDimension(1));
  int bitsPerComponent = image->GetPixelType().GetBitsPerComponent();
  int numberOfComponents = image->GetPixelType().GetNumberOfComponents();
  //expecting 8 bits per component, and 4 components, it might be more efficient to convert from RGBA to gray here ?
  cv::Mat cvImage;
  assert ( bitsPerComponent == 8 && (numberOfComponents == 3 || numberOfComponents == 4 || numberOfComponents == 1 ) );
  //we can't handle anything else
 // cvImage = cv::Mat(width, height, CV_8UC(numberOfComponents), const_cast<void*>(inputPointer), CV_AUTOSTEP);
 // CV_AUTOSTEP doesn't work
  cvImage = cv::Mat(height,width, CV_8UC(numberOfComponents), const_cast<void*>(inputPointer), width * bitsPerComponent/8 * numberOfComponents);

  return cvImage;
}

} // namespace

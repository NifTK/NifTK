/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkOpenCVImageConversion_h
#define niftkOpenCVImageConversion_h

#include "niftkOpenCVImageConversionExports.h"
#include <mitkImage.h>
#include <cv.h>

namespace niftk
{

/**
* Supports RGB, RGBA and grayscale images, currently 8-bit per channel only!.
* Known bug: does not take care of different channel layouts: BGR vs RGB!
*/
mitk::Image::Pointer NIFTKOPENCVIMAGECONVERSION_EXPORT CreateMitkImage(const IplImage* image);

/**
* Same as above but takes cv:Mat
*/
mitk::Image::Pointer NIFTKOPENCVIMAGECONVERSION_EXPORT CreateMitkImage(const cv::Mat* image);

/**
* mitk::Image::Pointer to cv::Mat* , supports 8 bit per channel RGB, RGBA and gray
* Known bug: does not take care of different channel layouts: BGR vs RGB!
*/
cv::Mat NIFTKOPENCVIMAGECONVERSION_EXPORT MitkImageToOpenCVMat ( const mitk::Image::Pointer );

} // namespace

#endif // niftkOpenCVImageConversion_h

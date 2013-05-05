/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "../Conversion/ImageConversion.h"
#include <mitkTestingMacros.h>
#include <mitkImageReadAccessor.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <boost/gil/gil_all.hpp>

// FIXME: how do i pull in non-test-driver code?
#include "ImageTestHelper.cxx"


//-----------------------------------------------------------------------------
static boost::gil::rgb8_image_t CreateTestImageRGB()
{
  boost::gil::rgb8_image_t    srcRgb(200, 100, 64);
  for (int y = 0; y < srcRgb.height(); ++y)
  {
    for (int x = 0; x < srcRgb.width(); ++x)
    {
      boost::gil::rgb8_pixel_t  p(x, y, std::min(x, y));
      boost::gil::view(srcRgb)(x, y) = p;
    }
  }
  return srcRgb;
}


//-----------------------------------------------------------------------------
static void RGBTest()
{
  // what do we expect from ipl->mitk image conversion?
  // 1) disconnect the mitk image from any buffers associated with ipl
  // 2) pitch-vs-width variations work
  // 3) rgb, rgba channel formats work


  mitk::Image::Pointer  outputImg;

  // scoping to trigger destructor
  {
    // FIXME: test what alignment param actually means!
    boost::gil::rgb8_image_t    srcRgb = CreateTestImageRGB();

    IplImage  srcIpl;
    cvInitImageHeader(&srcIpl, cvSize(srcRgb.width(), srcRgb.height()), IPL_DEPTH_8U, 3);
    cvSetData(&srcIpl, (void*) &boost::gil::view(srcRgb)(0, 0)[0], (char*) &boost::gil::view(srcRgb)(0, 1)[0] - (char*) &boost::gil::view(srcRgb)(0, 0)[0]);

    outputImg = niftk::CreateMitkImage(&srcIpl);
    MITK_TEST_CONDITION_REQUIRED(outputImg.IsNotNull(), "CreateMitkImage() returns non-null");
  }

  // build a gil view out of the mitk image buffers
  mitk::ImageReadAccessor  imgReadAccess(outputImg);
  const void* imgPtr            = imgReadAccess.GetData();
  int         width             = outputImg->GetDimension(0);
  int         height            = outputImg->GetDimension(1);
  int         numComponents     = outputImg->GetPixelType().GetNumberOfComponents();
  int         bitsPerComponent  = outputImg->GetPixelType().GetBitsPerComponent();
  MITK_TEST_CONDITION(width == 200, "MITK image has correct width (200)");
  MITK_TEST_CONDITION(height == 100, "MITK image has correct height (100)");
  MITK_TEST_CONDITION(outputImg->GetDimension(2) == 1, "MITK image has correct depth (1)");
  MITK_TEST_CONDITION(numComponents == 3, "MITK image has 3 components");
  MITK_TEST_CONDITION(bitsPerComponent == 8, "MITK image has 8 bits per component");

  boost::gil::rgb8_view_t   checkView = boost::gil::interleaved_view(
    width, height, (boost::gil::rgb8_pixel_t*) imgPtr, width * numComponents * (bitsPerComponent / 8));

  bool imagesAreTheSame = AreImagesTheSame<boost::gil::rgb8_pixel_t, boost::gil::rgb8_pixel_t>(boost::gil::const_view(CreateTestImageRGB()), checkView);
  MITK_TEST_CONDITION(imagesAreTheSame, "MITK image is the same as test image");
}


//-----------------------------------------------------------------------------
int ImageConversionTest(int /*argc*/, char* /*argv*/[])
{
  // check whether testing code works
  TestAreImagesTheSame();

  RGBTest();

  return EXIT_SUCCESS;
}

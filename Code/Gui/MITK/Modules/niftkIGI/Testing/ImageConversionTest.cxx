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
static boost::gil::rgba8_image_t CreateTestImageRGBA()
{
  boost::gil::rgba8_image_t    srcRgba(213, 165, 64);
  for (int y = 0; y < srcRgba.height(); ++y)
  {
    for (int x = 0; x < srcRgba.width(); ++x)
    {
      boost::gil::rgba8_pixel_t  p(x, y, std::min(x, y), std::max(x, y));
      boost::gil::view(srcRgba)(x, y) = p;
    }
  }
  return srcRgba;
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
    boost::gil::rgb8_image_t    srcRgb = CreateTestImageRGB();

    IplImage  srcIpl;
    cvInitImageHeader(&srcIpl, cvSize(srcRgb.width(), srcRgb.height()), IPL_DEPTH_8U, 3);
    cvSetData(&srcIpl, (void*) &boost::gil::view(srcRgb)(0, 0)[0], (char*) &boost::gil::view(srcRgb)(0, 1)[0] - (char*) &boost::gil::view(srcRgb)(0, 0)[0]);

    outputImg = niftk::CreateMitkImage(&srcIpl);
    MITK_TEST_CONDITION_REQUIRED(outputImg.IsNotNull(), "CreateMitkImage() [RGB] returns non-null");
  }

  // build a gil view out of the mitk image buffers
  mitk::ImageReadAccessor  imgReadAccess(outputImg);
  const void* imgPtr            = imgReadAccess.GetData();
  int         width             = outputImg->GetDimension(0);
  int         height            = outputImg->GetDimension(1);
  int         numComponents     = outputImg->GetPixelType().GetNumberOfComponents();
  int         bitsPerComponent  = outputImg->GetPixelType().GetBitsPerComponent();
  MITK_TEST_CONDITION(width == 200, "MITK image [RGB] has correct width (200)");
  MITK_TEST_CONDITION(height == 100, "MITK image [RGB] has correct height (100)");
  MITK_TEST_CONDITION(outputImg->GetDimension(2) == 1, "MITK image [RGB] has correct depth (1)");
  MITK_TEST_CONDITION(numComponents == 3, "MITK image [RGB] has 3 components");
  MITK_TEST_CONDITION(bitsPerComponent == 8, "MITK image [RGB] has 8 bits per component");

  boost::gil::rgb8_view_t   checkView = boost::gil::interleaved_view(
    width, height, (boost::gil::rgb8_pixel_t*) imgPtr, width * numComponents * (bitsPerComponent / 8));

  bool imagesAreTheSame = AreImagesTheSame<boost::gil::rgb8_pixel_t, boost::gil::rgb8_pixel_t>(boost::gil::const_view(CreateTestImageRGB()), checkView);
  MITK_TEST_CONDITION(imagesAreTheSame, "MITK image [RGB] is the same as test image");
}


//-----------------------------------------------------------------------------
// this is a copy-n-paste of RGBTest()
static void RGBATest()
{
  mitk::Image::Pointer  outputImg;

  // scoping to trigger destructor
  {
    boost::gil::rgba8_image_t    srcRgba = CreateTestImageRGBA();

    IplImage  srcIpl;
    cvInitImageHeader(&srcIpl, cvSize(srcRgba.width(), srcRgba.height()), IPL_DEPTH_8U, 4);
    cvSetData(&srcIpl, (void*) &boost::gil::view(srcRgba)(0, 0)[0], (char*) &boost::gil::view(srcRgba)(0, 1)[0] - (char*) &boost::gil::view(srcRgba)(0, 0)[0]);

    outputImg = niftk::CreateMitkImage(&srcIpl);
    MITK_TEST_CONDITION_REQUIRED(outputImg.IsNotNull(), "CreateMitkImage() [RGBA] returns non-null");
  }

  // build a gil view out of the mitk image buffers
  mitk::ImageReadAccessor  imgReadAccess(outputImg);
  const void* imgPtr            = imgReadAccess.GetData();
  int         width             = outputImg->GetDimension(0);
  int         height            = outputImg->GetDimension(1);
  int         numComponents     = outputImg->GetPixelType().GetNumberOfComponents();
  int         bitsPerComponent  = outputImg->GetPixelType().GetBitsPerComponent();
  MITK_TEST_CONDITION(width == 213, "MITK image [RGBA] has correct width (213)");
  MITK_TEST_CONDITION(height == 165, "MITK image [RGBA] has correct height (165)");
  MITK_TEST_CONDITION(outputImg->GetDimension(2) == 1, "MITK image [RGBA] has correct depth (1)");
  MITK_TEST_CONDITION(numComponents == 4, "MITK image [RGBA] has 4 components");
  MITK_TEST_CONDITION(bitsPerComponent == 8, "MITK image [RGBA] has 8 bits per component");

  boost::gil::rgba8_view_t   checkView = boost::gil::interleaved_view(
    width, height, (boost::gil::rgba8_pixel_t*) imgPtr, width * numComponents * (bitsPerComponent / 8));

  bool imagesAreTheSame = AreImagesTheSame<boost::gil::rgba8_pixel_t, boost::gil::rgba8_pixel_t>(boost::gil::const_view(CreateTestImageRGBA()), checkView);
  MITK_TEST_CONDITION(imagesAreTheSame, "MITK image [RGBA] is the same as test image");
}


//-----------------------------------------------------------------------------
int ImageConversionTest(int /*argc*/, char* /*argv*/[])
{
  // check whether testing code works
  TestAreImagesTheSame();

  RGBTest();
  RGBATest();

  return EXIT_SUCCESS;
}

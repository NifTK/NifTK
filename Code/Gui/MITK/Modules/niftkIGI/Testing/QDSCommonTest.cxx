/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <mitkTestingMacros.h>
#include "QDSCommon.h"
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc_c.h>

// FIXME: how do i pull in non-test-driver code?
#include "ImageTestHelper.cxx"


//-----------------------------------------------------------------------------
template <typename P>
void MakeGradientImage(typename boost::gil::image<P, false>::view_t dst)
{
  // this is like an integral image
  boost::gil::fill_pixels(dst, P());
  for (int y = 0; y < dst.height(); ++y)
  {
    for (int x = 0; x < dst.width(); ++x)
    {
      unsigned int  sum = 0;
      // extremely naive implementation
      for (int j = 0; j < y; ++j)
      {
        for (int i = 0; i < x; ++i)
        {
          sum += dst;
        }
      }

      dst(x, y) = sum;
    }
  }
}


//-----------------------------------------------------------------------------
void BuildTextureDescriptorTests()
{
  boost::gil::gray8_image_t   tiny(0, 0);
  boost::gil::gray8_image_t   small(10, 10);
  boost::gil::gray8_image_t   big(100, 100);

  // we'd except an error if input and output size is different
  // (the mitk test macros for exceptions are a bit annoying)
  try
  {
    niftk::BuildTextureDescriptor(boost::gil::const_view(small), boost::gil::view(big));
    MITK_TEST_CONDITION(!"No exception thrown", "BuildTextureDescriptor: Exception on invalid img size");
  }
  catch (const std::runtime_error& e)
  {
    MITK_TEST_CONDITION("Threw and caught correct exception", "BuildTextureDescriptor: Exception on invalid img size");
  }
  catch (...)
  {
    MITK_TEST_CONDITION(!"Threw wrong exception", "BuildTextureDescriptor: Exception on invalid img size");
  }


  // zero sized image should not break
  try
  {
    niftk::BuildTextureDescriptor(boost::gil::const_view(tiny), boost::gil::view(tiny));
    MITK_TEST_CONDITION("No exception thrown", "BuildTextureDescriptor: zero-sized image does not break");
  }
  catch (...)
  {
    MITK_TEST_CONDITION(!"Exception thrown", "BuildTextureDescriptor: zero-sized image does not break");
  }


  // for an image without any features we'd expect zero-value output
  boost::gil::gray8_image_t   small_output(small.dimensions());
  boost::gil::fill_pixels(boost::gil::view(small_output), boost::gil::gray8_pixel_t(255));
  boost::gil::fill_pixels(boost::gil::view(small), boost::gil::gray8_pixel_t(0));
  niftk::BuildTextureDescriptor(boost::gil::const_view(small), boost::gil::view(small_output));
  bool  nonzerooutput = false;
  for (int y = 0; y < small_output.height(); ++y)
  {
    for (int x = 0; x < small_output.width(); ++x)
    {
      if (boost::gil::const_view(small_output)(x, y)[0] != 0)
      {
        nonzerooutput = true;
      }
    }
  }
  MITK_TEST_CONDITION(nonzerooutput == false, "BuildTextureDescriptor: feature-less input produces zero-value output");
}


//-----------------------------------------------------------------------------
void calcIntegralImages(boost::gil::gray8c_view_t img, boost::gil::gray32s_view_t integral, boost::gil::gray64f_view_t squared)
{
  IplImage  grayipl;
  cvInitImageHeader(&grayipl, cvSize(img.width(), img.height()), IPL_DEPTH_8U, 1);
  cvSetData(&grayipl, (void*) &img(0, 0)[0], (char*) &img(0, 1)[0] - (char*) &img(0, 0)[0]);

  IplImage  integralipl;
  cvInitImageHeader(&integralipl, cvSize(integral.width(), integral.height()), IPL_DEPTH_32S, 1);
  cvSetData(&integralipl, &integral(0, 0)[0], (char*) &integral(0, 1)[0] - (char*) &integral(0, 0)[0]);

  IplImage  squaredintegralipl;
  cvInitImageHeader(&squaredintegralipl, cvSize(squared.width(), squared.height()), IPL_DEPTH_64F, 1);
  cvSetData(&squaredintegralipl, &squared(0, 0)[0], (char*) &squared(0, 1)[0] - (char*) &squared(0, 0)[0]);

  cvIntegral(&grayipl,  &integralipl,  &squaredintegralipl);
}


//-----------------------------------------------------------------------------
void Zncc_C1Tests()
{
  // pre-requisite for zncc to work is that opencv generates the correct integral image
  // question is: was this "inclusive" or "exclusive" scan... cant remember...

  // we start with a uniformly gray image.
  boost::gil::gray8_image_t   uniformgray(100, 100);
  boost::gil::fill_pixels(boost::gil::view(uniformgray), boost::gil::gray8_pixel_t(1));
//boost::gil::gray32s_image_t   expectedintegral(100+1, 100+1);
  boost::gil::gray32s_image_t   grayopencvintegral(uniformgray.width() + 1, uniformgray.height() + 1);
  boost::gil::gray64f_image_t   grayopencvsquaredintegral(grayopencvintegral.dimensions());


  boost::gil::gray8_image_t   uniformblack(uniformgray.dimensions());
  boost::gil::fill_pixels(boost::gil::view(uniformblack), boost::gil::gray8_pixel_t(0));
  boost::gil::gray32s_image_t   blackopencvintegral(uniformblack.width() + 1, uniformblack.height() + 1);
  boost::gil::gray64f_image_t   blackopencvsquaredintegral(blackopencvintegral.dimensions());

  calcIntegralImages(boost::gil::const_view(uniformgray), boost::gil::view(grayopencvintegral), boost::gil::view(grayopencvsquaredintegral));
  calcIntegralImages(boost::gil::const_view(uniformblack), boost::gil::view(blackopencvintegral), boost::gil::view(blackopencvsquaredintegral));


  // check that our test method works
//MITK_TEST_CONDITION_REQUIRED(
//  (!AreImagesTheSame<boost::gil::gray8c_pixel_t, boost::gil::gray32sc_pixel_t>
//    (boost::gil::const_view(uniformgray), boost::gil::const_view(expectedintegral))), 
//    "Zncc_C1: pre-requisites: our test equipment works");

  // FIXME: come up with a way to check it!
//MITK_TEST_CONDITION(
//  (!AreImagesTheSame<boost::gil::gray32sc_pixel_t, boost::gil::gray32sc_pixel_t>
//    (boost::gil::const_view(expectedintegral), boost::gil::const_view(opencvintegral))), 
//    "Zncc_C1: pre-requisites: integral image is what we expect");


  boost::gil::gray8_image_t   rampleftright(100, 100);
  for (int y = 0; y < rampleftright.height(); ++y)
  {
    for (int x = 0; x < rampleftright.width(); ++x)
    {
      boost::gil::view(rampleftright)(x, y)[0] = x;
    }
  }
  boost::gil::gray32s_image_t   rampleftrightintegral(rampleftright.width() + 1, rampleftright.height() + 1);
  boost::gil::gray64f_image_t   rampleftrightsquaredintegral(rampleftrightintegral.dimensions());
  calcIntegralImages(boost::gil::const_view(rampleftright), boost::gil::view(rampleftrightintegral), boost::gil::view(rampleftrightsquaredintegral));

  boost::gil::gray8_image_t   ramprightleft(100, 100);
  for (int y = 0; y < ramprightleft.height(); ++y)
  {
    for (int x = 0; x < ramprightleft.width(); ++x)
    {
      boost::gil::view(ramprightleft)(x, y)[0] = 255 - x;
    }
  }
  boost::gil::gray32s_image_t   ramprightleftintegral(ramprightleft.width() + 1, ramprightleft.height() + 1);
  boost::gil::gray64f_image_t   ramprightleftsquaredintegral(ramprightleftintegral.dimensions());
  calcIntegralImages(boost::gil::const_view(ramprightleft), boost::gil::view(ramprightleftintegral), boost::gil::view(ramprightleftsquaredintegral));


  // cross-correlation with itself
  float uniform_zncc = niftk::Zncc_C1(
                          uniformgray.width() / 2, uniformgray.height() / 2, 
                          uniformgray.width() / 2, uniformgray.height() / 2, 3, 
                          boost::gil::const_view(uniformgray), boost::gil::const_view(uniformgray), 
                          boost::gil::const_view(grayopencvintegral), boost::gil::const_view(grayopencvintegral), 
                          boost::gil::const_view(grayopencvsquaredintegral), boost::gil::const_view(grayopencvsquaredintegral));
  MITK_TEST_CONDITION((std::abs(1.0f - uniform_zncc) < 0.0001f), "Zncc_C1: correlation with itself");

  // ...big window
  uniform_zncc = niftk::Zncc_C1(
                          uniformgray.width() / 2, uniformgray.height() / 2, 
                          uniformgray.width() / 2, uniformgray.height() / 2, 15, 
                          boost::gil::const_view(uniformgray), boost::gil::const_view(uniformgray), 
                          boost::gil::const_view(grayopencvintegral), boost::gil::const_view(grayopencvintegral), 
                          boost::gil::const_view(grayopencvsquaredintegral), boost::gil::const_view(grayopencvsquaredintegral));
  MITK_TEST_CONDITION((std::abs(1.0f - uniform_zncc) < 0.0001f), "Zncc_C1: correlation with itself (big window)");

  uniform_zncc = niftk::Zncc_C1(
                          rampleftright.width() / 2, rampleftright.height() / 2, 
                          rampleftright.width() / 2, rampleftright.height() / 2, 3, 
                          boost::gil::const_view(rampleftright), boost::gil::const_view(rampleftright), 
                          boost::gil::const_view(rampleftrightintegral), boost::gil::const_view(rampleftrightintegral), 
                          boost::gil::const_view(rampleftrightsquaredintegral), boost::gil::const_view(rampleftrightsquaredintegral));
  MITK_TEST_CONDITION((std::abs(1.0f - uniform_zncc) < 0.0001f), "Zncc_C1: correlation with itself");

  // ...big window
  uniform_zncc = niftk::Zncc_C1(
                          rampleftright.width() / 2, rampleftright.height() / 2, 
                          rampleftright.width() / 2, rampleftright.height() / 2, 15, 
                          boost::gil::const_view(rampleftright), boost::gil::const_view(rampleftright), 
                          boost::gil::const_view(rampleftrightintegral), boost::gil::const_view(rampleftrightintegral), 
                          boost::gil::const_view(rampleftrightsquaredintegral), boost::gil::const_view(rampleftrightsquaredintegral));
  MITK_TEST_CONDITION((std::abs(1.0f - uniform_zncc) < 0.0001f), "Zncc_C1: correlation with itself (big window)");

  // cross-correlation with an offset
  uniform_zncc = niftk::Zncc_C1(
                          uniformgray.width() / 2, uniformgray.height() / 2, 
                          uniformblack.width() / 2, uniformblack.height() / 2, 3, 
                          boost::gil::const_view(uniformgray), boost::gil::const_view(uniformblack), 
                          boost::gil::const_view(grayopencvintegral), boost::gil::const_view(blackopencvintegral), 
                          boost::gil::const_view(grayopencvsquaredintegral), boost::gil::const_view(blackopencvsquaredintegral));
  MITK_TEST_CONDITION((std::abs(1.0f - uniform_zncc) < 0.0001f), "Zncc_C1: correlation with offset");

  uniform_zncc = niftk::Zncc_C1(
                          uniformgray.width() / 2, uniformgray.height() / 2, 
                          uniformblack.width() / 2, uniformblack.height() / 2, 15, 
                          boost::gil::const_view(uniformgray), boost::gil::const_view(uniformblack), 
                          boost::gil::const_view(grayopencvintegral), boost::gil::const_view(blackopencvintegral), 
                          boost::gil::const_view(grayopencvsquaredintegral), boost::gil::const_view(blackopencvsquaredintegral));
  MITK_TEST_CONDITION((std::abs(1.0f - uniform_zncc) < 0.0001f), "Zncc_C1: correlation with offset (big window)");

  // cross-correlation with opposing ramps
  uniform_zncc = niftk::Zncc_C1(
                          rampleftright.width() / 2, rampleftright.height() / 2, 
                          ramprightleft.width() / 2, ramprightleft.height() / 2, 3, 
                          boost::gil::const_view(rampleftright), boost::gil::const_view(ramprightleft), 
                          boost::gil::const_view(rampleftrightintegral), boost::gil::const_view(ramprightleftintegral), 
                          boost::gil::const_view(rampleftrightsquaredintegral), boost::gil::const_view(ramprightleftsquaredintegral));
  MITK_TEST_CONDITION((std::abs(-1.0f - uniform_zncc) < 0.0001f), "Zncc_C1: correlation of inverted img");

  uniform_zncc = niftk::Zncc_C1(
                          rampleftright.width() / 2, rampleftright.height() / 2, 
                          ramprightleft.width() / 2, ramprightleft.height() / 2, 15, 
                          boost::gil::const_view(rampleftright), boost::gil::const_view(ramprightleft), 
                          boost::gil::const_view(rampleftrightintegral), boost::gil::const_view(ramprightleftintegral), 
                          boost::gil::const_view(rampleftrightsquaredintegral), boost::gil::const_view(ramprightleftsquaredintegral));
  MITK_TEST_CONDITION((std::abs(-1.0f - uniform_zncc) < 0.0001f), "Zncc_C1: correlation of inverted img (big window)");
}


//-----------------------------------------------------------------------------
int QDSCommonTest(int /*argc*/, char* /*argv*/[])
{
  TestAreImagesTheSame();

  BuildTextureDescriptorTests();

  Zncc_C1Tests();


  return EXIT_SUCCESS;
}

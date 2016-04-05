/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include <boost/gil/gil_all.hpp>
#include <boost/typeof/typeof.hpp>


//-----------------------------------------------------------------------------
// FIXME: the pixel type template parameters are not nice to use
template <typename A, typename B>
bool AreImagesTheSame(const typename boost::gil::image<A, false>::const_view_t& a, const typename boost::gil::image<B, false>::const_view_t& b)
{
  if (a.dimensions() != b.dimensions())
    return false;

  // FIXME: check number of channels

  for (int y = 0; y < a.height(); ++y)
  {
    for (int x = 0; x < a.width(); ++x)
    {
      BOOST_AUTO(ap, a(x, y));
      BOOST_AUTO(bp, b(x, y));

      if (ap != bp)
      {
        return false;
      }
    }
  }

  return true;
}


//-----------------------------------------------------------------------------
static void TestAreImagesTheSameRGB()
{
  boost::gil::rgb8_image_t  zero(123, 45);
  boost::gil::rgb8_image_t  one(zero.dimensions());
  boost::gil::rgb8_image_t  wrongsize(45, 123);
  boost::gil::fill_pixels(boost::gil::view(zero), boost::gil::rgb8_pixel_t(0, 0, 0));
  boost::gil::fill_pixels(boost::gil::view(one), boost::gil::rgb8_pixel_t(1, 1, 1));
  boost::gil::fill_pixels(boost::gil::view(wrongsize), boost::gil::rgb8_pixel_t(0, 0, 0));

  MITK_TEST_CONDITION(
    (AreImagesTheSame<boost::gil::rgb8_pixel_t, boost::gil::rgb8_pixel_t>(boost::gil::const_view(zero), boost::gil::const_view(zero))), 
    "Checking test code: RGB image is self-same");
  MITK_TEST_CONDITION(
    !(AreImagesTheSame<boost::gil::rgb8_pixel_t, boost::gil::rgb8_pixel_t>(boost::gil::const_view(one), boost::gil::const_view(zero))), 
    "Checking test code: different RGB images are not the same");
  MITK_TEST_CONDITION(
    !(AreImagesTheSame<boost::gil::rgb8_pixel_t, boost::gil::rgb8_pixel_t>(boost::gil::const_view(wrongsize), boost::gil::const_view(zero))), 
    "Checking test code: differently sized RGB images are not the same");
}


//-----------------------------------------------------------------------------
static void TestAreImagesTheSameRGBA()
{
  boost::gil::rgba8_image_t  zero(123, 45);
  boost::gil::rgba8_image_t  one(zero.dimensions());
  boost::gil::rgba8_image_t  wrongsize(45, 123);
  boost::gil::fill_pixels(boost::gil::view(zero), boost::gil::rgba8_pixel_t(0, 0, 0, 0));
  boost::gil::fill_pixels(boost::gil::view(one), boost::gil::rgba8_pixel_t(1, 1, 1, 1));
  boost::gil::fill_pixels(boost::gil::view(wrongsize), boost::gil::rgba8_pixel_t(0, 0, 0, 0));

  MITK_TEST_CONDITION(
    (AreImagesTheSame<boost::gil::rgba8_pixel_t, boost::gil::rgba8_pixel_t>(boost::gil::const_view(zero), boost::gil::const_view(zero))), 
    "Checking test code: RGBA image is self-same");
  MITK_TEST_CONDITION(
    !(AreImagesTheSame<boost::gil::rgba8_pixel_t, boost::gil::rgba8_pixel_t>(boost::gil::const_view(one), boost::gil::const_view(zero))), 
    "Checking test code: different RGBA images are not the same");
  MITK_TEST_CONDITION(
    !(AreImagesTheSame<boost::gil::rgba8_pixel_t, boost::gil::rgba8_pixel_t>(boost::gil::const_view(wrongsize), boost::gil::const_view(zero))), 
    "Checking test code: differently sized RGBA images are not the same");
}


//-----------------------------------------------------------------------------
// GIL makes sure compilation fails if we are comparing different pixel types.
// So we do not need this method. But leave it here for now to document that
// we thought about this case.
static void TestAreImagesWithDifferingTypes()
{
  // same size/content but different types
  boost::gil::rgba8_image_t  zero1(123, 45);
  boost::gil::rgb8_image_t   zero2(zero1.dimensions());
  // not actually zero... doesnt matter
  // but explicit rgba alpha has same value as non-existing implicit rgb alpha: zero
  boost::gil::fill_pixels(boost::gil::view(zero1), boost::gil::rgba8_pixel_t(3, 7, 9, 0));
  boost::gil::fill_pixels(boost::gil::view(zero2), boost::gil::rgb8_pixel_t(3, 7, 9));

//  MITK_TEST_CONDITION(
//    !(AreImagesTheSame<boost::gil::rgba8_pixel_t, boost::gil::rgb8_pixel_t>(boost::gil::const_view(zero1), boost::gil::const_view(zero2))), 
//    "Checking test code: similar RGB and RGBA images are not the same");


  // same number of components (and swizzled content) but different layout
  boost::gil::bgr8_image_t  zero3(zero2.dimensions());
  boost::gil::fill_pixels(boost::gil::view(zero3), boost::gil::rgb8_pixel_t(3, 7, 9));

//  MITK_TEST_CONDITION(
//    !(AreImagesTheSame<boost::gil::rgb8_pixel_t, boost::gil::bgr8_pixel_t>(boost::gil::const_view(zero2), boost::gil::const_view(zero3))), 
//    "Checking test code: similar RGB and BGR images are not the same");
}


//-----------------------------------------------------------------------------
static void TestAreImagesTheSame()
{
  TestAreImagesTheSameRGB();
  TestAreImagesTheSameRGBA();
  TestAreImagesWithDifferingTypes();
}

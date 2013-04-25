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



int QDSCommonTest(int /*argc*/, char* /*argv*/[])
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

  return EXIT_SUCCESS;
}

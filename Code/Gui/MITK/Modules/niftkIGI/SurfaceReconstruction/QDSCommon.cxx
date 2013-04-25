/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "QDSCommon.h"
#include <boost/gil/gil_all.hpp>

#ifdef _OMP
#include <omp.h>
#endif


namespace niftk
{


//-----------------------------------------------------------------------------
void BuildTextureDescriptor(const boost::gil::gray8c_view_t src, const boost::gil::gray8_view_t dst)
{
  if (src.dimensions() != dst.dimensions())
  {
    throw std::runtime_error("Input image dimensions does not match output image");
  }

  // filled only to put a defined value along the image edge which is not processed by the loop below.
  // this shouldnt be necessary though because the propagation stays away from the border anyway.
  boost::gil::fill_pixels(dst, boost::gil::gray8_pixel_t(0));

  // no point running more than 2 threads here, loop is very simple to start with.
  // with more threads we'll just end up with cache thrash, etc.
  #pragma omp parallel for num_threads(2)
  for (int y = 1; y < src.height() - 1; ++y)
  {
    for (int x = 1; x < src.width() - 1; ++x)
    {
      const boost::gil::gray8c_pixel_t& pixel = src(x, y);

      //difference
      int a = std::abs(pixel - src(x - 1, y));
      int b = std::abs(pixel - src(x + 1, y));
      int c = std::abs(pixel - src(x, y - 1));
      int d = std::abs(pixel - src(x, y + 1));

      // FIXME: dan had this as max(), but min() would make more sense?
      int val = std::max(a, std::max(b, std::max(c, d)));

      // clamp
      // FIXME: dont know when or if this happens!
      if (val > 255)
      {
        assert(false);
        val = 255;
      }
      if (val < 0)
      {
        assert(false);
        val = 0;
      }

      dst(x, y) = val;
    }
  }
}


//-----------------------------------------------------------------------------
// NOTE: returns values in the range of [-1...+1]
// FIXME: this needs proper unit-testing! desperately!
float Zncc_C1(int p0x, int p0y, int p1x, int p1y, int w, boost::gil::gray8c_view_t img0, boost::gil::gray8c_view_t img1, boost::gil::gray32sc_view_t integral0, boost::gil::gray32sc_view_t integral1, boost::gil::gray64fc_view_t square0, boost::gil::gray64fc_view_t square1)
{
  // random variables used by code below
  // this is a relic from dan's code
  const unsigned char*  data0 = &img0(0, 0)[0];
  const unsigned char*  data1 = &img1(0, 0)[0];
  const int*            sum0  = &integral0(0, 0)[0];
  const int*            sum1  = &integral1(0, 0)[0];
  const double*         ssum0 = (double*) &square0(0, 0)[0];
  const double*         ssum1 = (double*) &square1(0, 0)[0];

  int   Step  = &img0(0, 1)[0] - &img0(0, 0)[0];
  int   Steps = &integral0(0, 1)[0] - &integral0(0, 0)[0];

  int x,y,otl,otr,obl,obr;
  double  m0 = 0.0,
          m1 = 0.0,
          s0 = 0.0,
          s1 = 0.0;

  const float wa = (2 * w + 1) * (2 * w + 1);

  int boy0 = (p0y - w) * Step + (p0x - w);
  int boy1 = (p1y - w) * Step + (p1x - w);

  int oy0=boy0,
      oy1=boy1;
  int ox0=0,
      ox1=0;

  // offsets for corners top-left, top-right, bottom-left, bottom-right
  int   w1 = w + 1;

  // offsets for left image
  otl = (p0y -  w) * Steps + (p0x - w);
  otr = (p0y -  w) * Steps + (p0x + w1);
  obl = (p0y + w1) * Steps + (p0x - w);
  obr = (p0y + w1) * Steps + (p0x + w1);

  // sum and squared sum for left window
  m0 = ((sum0[obr] +  sum0[otl]) - ( sum0[obl] +  sum0[otr]));
  s0 = (ssum0[obr] + ssum0[otl]) - (ssum0[obl] + ssum0[otr]);

  // offsets for right image
  otl = (p1y -  w) * Steps + (p1x - w);
  otr = (p1y -  w) * Steps + (p1x + w1);
  obl = (p1y + w1) * Steps + (p1x - w);
  obr = (p1y + w1) * Steps + (p1x + w1);

  // sum and squares sum for right window
  m1 = ((sum1[obr] +  sum1[otl]) - ( sum1[obl] + sum1[otr]));
  s1 = (ssum1[obr] + ssum1[otl]) - (ssum1[obl] + ssum1[otr]);

  // window means
  m0 /= wa;
  m1 /= wa;

  // standard deviations
  s0 = std::sqrt(s0 - wa * m0 * m0);
  s1 = std::sqrt(s1 - wa * m1 * m1);

  // we need to check for these specifically otherwise the final eq blows up
  if ((s0 <= 0) || (s1 <= 0))
  {
    // if stddev is zero then we have flat-line input
    // if both are flat then we have perfect correlation
    if ((s0 <= 0) && (s1 <= 0))
    {
      return 1;
    }
    // otherwise, don't know actually...
    return 0;
  }

  float zncc = 0;
  for (y = -w; y <= w; ++y, oy1 += Step, oy0 += Step)
  {
    ox0 = 0;
    ox1 = 0;
    const unsigned char* line0 = &data0[oy0];
    const unsigned char* line1 = &data1[oy1];
    for (x = -w; x <= w; ++x)
    {
      zncc += (float) line0[ox0++] * (float) line1[ox1++];
    }
  }

  // the final result
  zncc = (zncc - wa * m0 * m1) / (s0 * s1);

//if (zncc < -1.0f)
//  std::cout << "burp";
//if (zncc > 1.0f)
//  std::cout << "bla";

  return zncc;
}


} // namespace

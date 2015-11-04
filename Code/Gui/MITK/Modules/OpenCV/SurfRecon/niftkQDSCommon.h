/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef niftkQDSCommon_h
#define niftkQDSCommon_h

#include "niftkOpenCVExports.h"
#include <boost/gil/gil_all.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core.hpp>

// the ancient version of boost that comes with mitk does not have
// a 64-bit floating point pixel format.
// so define this, based on my hacked boost version:
//  https://bitbucket.org/bruelltuete/boost/commits/27198c44596696d1bb1ae686c828efa82b08fd9f
namespace boost
{
namespace gil
{

struct double_zero { static float apply() { return 0.0; } };
struct double_one  { static float apply() { return 1.0; } };

typedef scoped_channel_value<double, double_zero, double_one> bits64f;

GIL_DEFINE_BASE_TYPEDEFS(64f,gray)

} // end namespace boost
} // end namespace gil

namespace niftk
{

struct NIFTKOPENCV_EXPORT RefPoint
{
  unsigned short  x;
  unsigned short  y;

  RefPoint(unsigned short _x = -1, unsigned short _y = -1)
    : x(_x), y(_y)
  {
  }

  RefPoint& operator=(const RefPoint& p)
  {
    x = p.x;
    y = p.y;
    return *this;
  }

  // conversion operator! exists for convinience with refmaps
  // FIXME: pixel assignment is done with a template parameter
  //        so this conversion is not actually called automagically!
  operator boost::gil::dev2n16_pixel_t() const
  {
    return boost::gil::dev2n16_pixel_t(x, y);
  }

  // compares coordinates, for use in priority_queue
  bool operator<(const RefPoint& rhs) const;
};


/**
* This is some kind of corner detector.
* If the output image has a low/zero value for a given pixel then the same pixel coordinate in your input
* image will have little to no texture suitable for tracking/matching.
*
* @throws std::runtime_error if src and dst dimensions are different
*/
void NIFTKOPENCV_EXPORT BuildTextureDescriptor(
    const boost::gil::gray8c_view_t src,
    const boost::gil::gray8_view_t dst);

float NIFTKOPENCV_EXPORT Zncc_C1(int p0x, int p0y, int p1x, int p1y, int w,
                                    boost::gil::gray8c_view_t img0,
                                    boost::gil::gray8c_view_t img1,
                                    boost::gil::gray32sc_view_t integral0,
                                    boost::gil::gray32sc_view_t integral1,
                                    boost::gil::gray64fc_view_t square0,
                                    boost::gil::gray64fc_view_t square1);

/**
* Base class for the (CPU-versions of) QDS stereo-matching.
*/
class NIFTKOPENCV_EXPORT QDSInterface
{
public:
  virtual ~QDSInterface();

  virtual void Process(const IplImage* left, const IplImage* right) = 0;

  virtual int GetWidth() const = 0;
  virtual int GetHeight() const = 0;

  virtual CvPoint GetMatch(int x, int y) const = 0;

  // caller needs to clean up
  virtual IplImage* CreateDisparityImage() const = 0;
};

} // namespace

#endif

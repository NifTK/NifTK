/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef QDSCommon_h
#define QDSCommon_h

#include "niftkIGIExports.h"
#include <boost/gil/gil_all.hpp>
#include <opencv2/core/types_c.h>


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

}
}


namespace niftk
{


/**
 * This is some kind of corner detector.
 * If the output image has a low/zero value for a given pixel then the same pixel coordinate in your input
 * image will have little to no texture suitable for tracking/matching.
 *
 * @throws std::runtime_error if src and dst dimensions are different
 */
void NIFTKIGI_EXPORT BuildTextureDescriptor(const boost::gil::gray8c_view_t src, const boost::gil::gray8_view_t dst);


float NIFTKIGI_EXPORT Zncc_C1(int p0x, int p0y, int p1x, int p1y, int w, boost::gil::gray8c_view_t img0, boost::gil::gray8c_view_t img1, boost::gil::gray32sc_view_t integral0, boost::gil::gray32sc_view_t integral1, boost::gil::gray64fc_view_t square0, boost::gil::gray64fc_view_t square1);


CvPoint3D32f triangulate(
    float p0x, float p0y, 
    const CvMat& intrinsic_left, const CvScalar& distortion_left,
    float p1x, float p1y, 
    const CvMat& intrinsic_right, const CvScalar& distortion_right,
    const CvMat& left2right_rotation, const CvMat& left2right_translation,
    float* err = 0
  );


} // namespace


#endif // QDSCommon_h

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
#include <opencv2/core/core.hpp>
#include <mitkCameraIntrinsics.h>


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


/**
 * Triangulates a pixel-pair in two views.
 * I've had problems with OpenCV's cvTriangulatePoints() in the past, hence our own implementation here.
 *
 * @param left2right_rotation a 3x3 matrix, row-major?
 * @param left2right_translation a 3x1 matrix (3 rows, 1 column)
 */
// FIXME: some of these overloads should go away! i just need to figure out first which ones are useful
CvPoint3D32f triangulate(
    float p0x, float p0y,
    const CvMat& intrinsic_left, const CvScalar& distortion_left,
    float p1x, float p1y,
    const CvMat& intrinsic_right, const CvScalar& distortion_right,
    const CvMat& left2right_rotation, const CvMat& left2right_translation,
    float* err = 0
  );
// overload for new opencv c++ types
CvPoint3D32f triangulate(
    float p0x, float p0y,
    const cv::Mat& intrinsic_left, const cv::Vec<float, 4>& distortion_left,
    float p1x, float p1y,
    const cv::Mat& intrinsic_right, const cv::Vec<float, 4>& distortion_right,
    const cv::Mat& left2right_rotation, const cv::Mat& left2right_translation,
    float* err = 0
  );
// for mitk stuff
CvPoint3D32f triangulate(
    float p0x, float p0y,
    mitk::CameraIntrinsics::Pointer intrinsic_left,
    float p1x, float p1y,
    mitk::CameraIntrinsics::Pointer intrinsic_right,
    const itk::Matrix<float, 4, 4>& left2right,//const cv::Mat& left2right_rotation, const cv::Mat& left2right_translation,
    float* err = 0
  );


} // namespace


#endif // QDSCommon_h

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
#include <boost/static_assert.hpp>

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


//-----------------------------------------------------------------------------
CvPoint3D32f triangulate(
    float p0x, float p0y, const CvMat& intrinsic_left,  const CvScalar& distortion_left,
    float p1x, float p1y, const CvMat& intrinsic_right, const CvScalar& distortion_right,
    const CvMat& left2right_rotation, const CvMat& left2right_translation,
    float* err
  )
{
  // FIXME: check that matrix type is float and has the correct size

  BOOST_STATIC_ASSERT((sizeof(CvPoint3D32f) == (3 * sizeof(float))));

  CvPoint3D32f  left = cvPoint3D32f
  (
    ((double) p0x - cvmGet(&intrinsic_left, 0, 2)/*principal_point*/) / cvmGet(&intrinsic_left, 0, 0)/*focal_length*/,
    ((double) p0y - cvmGet(&intrinsic_left, 1, 2)/*principal_point*/) / cvmGet(&intrinsic_left, 1, 1)/*focal_length*/,
    1
  );

  CvPoint3D32f  right = cvPoint3D32f
  (
    ((double) p1x - cvmGet(&intrinsic_right, 0, 2)/*principal_point*/) / cvmGet(&intrinsic_right, 0, 0)/*focal_length*/,
    ((double) p1y - cvmGet(&intrinsic_right, 1, 2)/*principal_point*/) / cvmGet(&intrinsic_right, 1, 1)/*focal_length*/,
    1
  );

#if 0
  std::pair<double, double> left_undist  = undistort(left[0],  left[1],  stereorig.left);
  std::pair<double, double> right_undist = undistort(right[0], right[1], stereorig.right);
    // just checking
  std::pair<double, double> left_dist  = distort(left_undist.first,  left_undist.second,  stereorig.left);
  std::pair<double, double> right_dist = distort(right_undist.first, right_undist.second, stereorig.right);

  left[0] = left_undist.first;
  left[1] = left_undist.second;
  right[0] = right_undist.first;
  right[1] = right_undist.second;
#endif

  // now to some messy stuff
  // this is based on code I got from Pete Mountney.

  double u[3];
  // FIXME: might have to flip col vs row here!
  u[0] = cvmGet(&left2right_rotation, 0, 0)/*stereorig.rotation[0]*/ * left.x + cvmGet(&left2right_rotation, 0, 1)/*stereorig.rotation[1]*/ * left.y + cvmGet(&left2right_rotation, 0, 2)/*stereorig.rotation[2]*/ * left.z;
  u[1] = cvmGet(&left2right_rotation, 1, 0)/*stereorig.rotation[3]*/ * left.x + cvmGet(&left2right_rotation, 1, 1)/*stereorig.rotation[4]*/ * left.y + cvmGet(&left2right_rotation, 1, 2)/*stereorig.rotation[5]*/ * left.z;
  u[2] = cvmGet(&left2right_rotation, 2, 0)/*stereorig.rotation[6]*/ * left.x + cvmGet(&left2right_rotation, 2, 1)/*stereorig.rotation[7]*/ * left.y + cvmGet(&left2right_rotation, 2, 2)/*stereorig.rotation[8]*/ * left.z;

  double n_nml = left.x * left.x + left.y * left.y + left.z * left.z;
  double n_nmr = right.x * right.x + right.y * right.y + right.z * right.z;

  double DD = n_nmr * n_nml - (u[0] * right.x + u[1] * right.y + u[2] * right.z) * (u[0] * right.x + u[1] * right.y + u[2] * right.z);

  // FIXME: column vs row vector style matrix!
  double  duT   =    u[0] * cvmGet(&left2right_translation, 0, 0)/*stereorig.translation[0]*/ +    u[1] * cvmGet(&left2right_translation, 0, 1)/*stereorig.translation[1]*/ +    u[2] * cvmGet(&left2right_translation, 0, 2)/*stereorig.translation[2]*/;
  double  dnmrT = right.x * cvmGet(&left2right_translation, 0, 0)/*stereorig.translation[0]*/ + right.y * cvmGet(&left2right_translation, 0, 1)/*stereorig.translation[1]*/ + right.z * cvmGet(&left2right_translation, 0, 2)/*stereorig.translation[2]*/;
  double  dunmr = right.x * u[0] + right.y * u[1] + right.z * u[2];

  double NN1 = dunmr * dnmrT - n_nmr * duT;
  double NN2 = n_nml * dnmrT - duT * dunmr;

  double Zl = NN1 / DD;
  double Zr = NN2 / DD;

  double X1[3];
  X1[0] = left.x * Zl;
  X1[1] = left.y * Zl;
  X1[2] = left.z * Zl;

  double X2[3];
  X2[0] = cvmGet(&left2right_rotation, 0, 0)/*stereorig.rotation[0]*/ * (right.x * Zr - cvmGet(&left2right_translation, 0, 0)/*stereorig.translation[0]*/) + cvmGet(&left2right_rotation, 1, 0)/*stereorig.rotation[3]*/ * (right.y * Zr - cvmGet(&left2right_translation, 0, 1)/*stereorig.translation[1]*/) + cvmGet(&left2right_rotation, 2, 0)/*stereorig.rotation[6]*/ * (right.z * Zr - cvmGet(&left2right_translation, 0, 2)/*stereorig.translation[2]*/);
  X2[1] = cvmGet(&left2right_rotation, 0, 1)/*stereorig.rotation[1]*/ * (right.x * Zr - cvmGet(&left2right_translation, 0, 0)/*stereorig.translation[0]*/) + cvmGet(&left2right_rotation, 1, 1)/*stereorig.rotation[4]*/ * (right.y * Zr - cvmGet(&left2right_translation, 0, 1)/*stereorig.translation[1]*/) + cvmGet(&left2right_rotation, 2, 1)/*stereorig.rotation[7]*/ * (right.z * Zr - cvmGet(&left2right_translation, 0, 2)/*stereorig.translation[2]*/);
  X2[2] = cvmGet(&left2right_rotation, 0, 2)/*stereorig.rotation[2]*/ * (right.x * Zr - cvmGet(&left2right_translation, 0, 0)/*stereorig.translation[0]*/) + cvmGet(&left2right_rotation, 1, 2)/*stereorig.rotation[5]*/ * (right.y * Zr - cvmGet(&left2right_translation, 0, 1)/*stereorig.translation[1]*/) + cvmGet(&left2right_rotation, 2, 2)/*stereorig.rotation[8]*/ * (right.z * Zr - cvmGet(&left2right_translation, 0, 2)/*stereorig.translation[2]*/);

  CvPoint3D32f  worldPointInLeftCam = cvPoint3D32f
  (
    (X1[0] + X2[0]) / 2.0,
    (X1[1] + X2[1]) / 2.0,
    (X1[2] + X2[2]) / 2.0
  );

  if (err != 0)
  {
    *err = std::sqrt(
        (worldPointInLeftCam.x - X1[0]) * (worldPointInLeftCam.x - X1[0]) +
        (worldPointInLeftCam.y - X1[1]) * (worldPointInLeftCam.y - X1[1]) +
        (worldPointInLeftCam.z - X1[2]) * (worldPointInLeftCam.z - X1[2]));
  }

  return worldPointInLeftCam;
}


//-----------------------------------------------------------------------------
CvPoint3D32f triangulate(
    float p0x, float p0y, const cv::Mat& intrinsic_left,  const cv::Vec<float, 4>& distortion_left,
    float p1x, float p1y, const cv::Mat& intrinsic_right, const cv::Vec<float, 4>& distortion_right,
    const cv::Mat& left2right_rotation, const cv::Mat& left2right_translation,
    float* err
  )
{
  return triangulate(
    p0x, p0y, (CvMat) intrinsic_left,  cvScalar(distortion_left[0],  distortion_left[1],  distortion_left[2],  distortion_left[3]),
    p1x, p1y, (CvMat) intrinsic_right, cvScalar(distortion_right[0], distortion_right[1], distortion_right[2], distortion_right[3]), 
    (CvMat) left2right_rotation, (CvMat) left2right_translation, err);
}


//-----------------------------------------------------------------------------
CvPoint3D32f triangulate(
    float p0x, float p0y, mitk::CameraIntrinsics::Pointer intrinsic_left,
    float p1x, float p1y, mitk::CameraIntrinsics::Pointer intrinsic_right,
    const itk::Matrix<float, 4, 4>& left2right,
    float* err
  )
{
  BOOST_STATIC_ASSERT((sizeof(mitk::Point4D) == sizeof(cv::Vec<float, 4>)));
  return triangulate(
    p0x, p0y, intrinsic_left->GetCameraMatrix(),  *((cv::Vec<float, 4>*) &intrinsic_left->GetDistorsionCoeffsAsPoint4D()),
    p1x, p1y, intrinsic_right->GetCameraMatrix(), *((cv::Vec<float, 4>*) &intrinsic_right->GetDistorsionCoeffsAsPoint4D()),
    cv::Mat(3, 3, CV_32F, (void*) left2right.GetVnlMatrix().data_block(), sizeof(float) * 4),
    cv::Mat(1, 3, CV_32F, (void*) left2right.GetVnlMatrix().data_block(), sizeof(float) * 4),
    err);
}


} // namespace

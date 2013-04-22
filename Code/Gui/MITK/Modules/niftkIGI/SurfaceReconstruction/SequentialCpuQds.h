/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#ifndef mitkSequentialCpuQds_h
#define mitkSequentialCpuQds_h

#include "niftkIGIExports.h"
#include <opencv2/core/types_c.h>
// FIXME: typedefs.hpp should be enough! but there seem to be some issues...
#include <boost/gil/typedefs.hpp>
#include <boost/gil/gil_all.hpp>
#include <vector>


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


namespace mitk 
{


struct PropagationParameters
{
  int   WinSizeX;   // similarity window, in pixels
  int   WinSizeY;

  int   BorderX;    // border in pixels to ignore
  int   BorderY;

  float Ct;         // correlation threshold
  float Tt;         // texture threshold
//float Et;         // epipolar threshold

  int   N;          // search neighbourhood size, in pixels
  int   Dg;         // disparity gradient threshold, in pixels
};


// FIXME: this is actually shared between all qds variants, so refactor this (later)
struct NIFTKIGI_EXPORT RefPoint
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


class NIFTKIGI_EXPORT SequentialCpuQds
{

public:
  SequentialCpuQds(int width, int height);
  ~SequentialCpuQds();


  // supports greyscale, RGB or RGBA images.
  // input dimensions need to match the width and height passed into constructor.
  void Process(const IplImage* left, const IplImage* right);


  int GetWidth() const;
  int GetHeight() const;


private:
  SequentialCpuQds(const SequentialCpuQds& copyme);
  SequentialCpuQds& operator=(const SequentialCpuQds& assignme);


private:
  void InitSparseFeatures();

  void QuasiDensePropagation();


private:
  // internal buffers are of fixed size
  int     m_Width;
  int     m_Height;

  PropagationParameters   m_PropagationParams;

  // internal buffers
  // grayscale version of the input passed to Process()
  boost::gil::gray8_image_t   m_LeftImg;
  boost::gil::gray8_image_t   m_RightImg;
  // integral image of the grayscale version
  boost::gil::gray32s_image_t m_LeftIntegral;
  boost::gil::gray32s_image_t m_RightIntegral;
  // squared-integral
  boost::gil::gray64f_image_t m_LeftSquaredIntegral;
  boost::gil::gray64f_image_t m_RightSquaredIntegral;
  // texture descriptor, i.e. cornerness image
  boost::gil::gray8_image_t   m_LeftTexture;
  boost::gil::gray8_image_t   m_RightTexture;
  // refmap: pixel coordinate into the other view
  boost::gil::dev2n16_image_t m_LeftRefMap;
  boost::gil::dev2n16_image_t m_RightRefMap;

  // these have no mem allocated themselves!
  // they only reference the gil images
  IplImage    m_LeftIpl;
  IplImage    m_RightIpl;
  IplImage    m_LeftIntegralIpl;
  IplImage    m_RightIntegralIpl;
  IplImage    m_LeftSquaredIntegralIpl;
  IplImage    m_RightSquaredIntegralIpl;

  static const int            NUM_MAX_FEATURES = 500;
  std::vector<CvPoint2D32f>   m_SparseFeaturesLeft;
  std::vector<CvPoint2D32f>   m_SparseFeaturesRight;
  std::vector<char>           m_FeatureStatus;

  // used only for disparity image debugging
  const int   m_MaxDisparity;
};

} // namespace

#endif // mitkSequentialCpuQds_h

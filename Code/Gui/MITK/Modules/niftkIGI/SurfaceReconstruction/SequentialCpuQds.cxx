/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "SequentialCpuQds.h"
#include "QDSCommon.h"
#include <queue>
#include <boost/gil/gil_all.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/video/tracking.hpp>
#include <boost/typeof/typeof.hpp>
#include <boost/static_assert.hpp>

#ifdef _OMP
#include <omp.h>
#endif


BOOST_STATIC_ASSERT((sizeof(niftk::RefPoint) == sizeof(boost::gil::dev2n16_pixel_t)));


namespace niftk 
{


//-----------------------------------------------------------------------------
bool RefPoint::operator<(const RefPoint& rhs) const 
{
  if (x < rhs.x)
    return true;
  if (x > rhs.x)
    return false;
  if (y < rhs.y)
    return true;
  if (y > rhs.y)
    return false;
  return false;
}


// Used to store match candidates in the priority queue
struct Match
{
  RefPoint  p0;
  RefPoint  p1;
  float     corr;

  bool operator<(const Match& rhs) const
  {
    return corr < rhs.corr;
  }
};


//-----------------------------------------------------------------------------
static bool CheckBorder(const Match& m, int bx, int by, int w, int h)
{
  assert(m.p0.x >= 0);
  assert(m.p0.y >= 0);
  assert(m.p1.x >= 0);
  assert(m.p1.y >= 0);

  if ((m.p0.x < bx    ) || 
      (m.p0.x > w - bx) || 
      (m.p0.y < by    ) || 
      (m.p0.y > h - by) || 
      (m.p1.x < bx    ) || 
      (m.p1.x > w - bx) || 
      (m.p1.y < by    ) || 
      (m.p1.y > h - by))
  {
    return false;
  }

  return true;
}


//-----------------------------------------------------------------------------
SequentialCpuQds::SequentialCpuQds(int width, int height)
  : m_Width(width), m_Height(height), m_MaxDisparity(std::max(width, height) / 7)
{
  m_LeftImg.recreate(width, height);
  m_RightImg.recreate(width, height);

  m_LeftIntegral.recreate(width + 1, height + 1);
  m_RightIntegral.recreate(width + 1, height + 1);
  m_LeftSquaredIntegral.recreate(width + 1, height + 1);
  m_RightSquaredIntegral.recreate(width + 1, height + 1);

  m_LeftTexture.recreate(width, height);
  m_RightTexture.recreate(width, height);

  m_LeftRefMap.recreate(width, height);
  m_RightRefMap.recreate(width, height);

  // the ipl images are used only for a few opencv calls
  // otherwise these simply reference the gil images on which all other processing happens
  cvInitImageHeader(&m_LeftIpl, cvSize(m_LeftImg.width(), m_LeftImg.height()), IPL_DEPTH_8U, 1);
  cvSetData(&m_LeftIpl, &boost::gil::view(m_LeftImg)(0, 0)[0], (char*) &boost::gil::view(m_LeftImg)(0, 1)[0] - (char*) &boost::gil::view(m_LeftImg)(0, 0)[0]);
  cvInitImageHeader(&m_RightIpl, cvSize(m_RightImg.width(), m_RightImg.height()), IPL_DEPTH_8U, 1);
  cvSetData(&m_RightIpl, &boost::gil::view(m_RightImg)(0, 0)[0], (char*) &boost::gil::view(m_RightImg)(0, 1)[0] - (char*) &boost::gil::view(m_RightImg)(0, 0)[0]);

  cvInitImageHeader(&m_LeftIntegralIpl, cvSize(m_LeftIntegral.width(), m_LeftIntegral.height()), IPL_DEPTH_32S, 1);
  cvSetData(&m_LeftIntegralIpl, &boost::gil::view(m_LeftIntegral)(0, 0)[0], (char*) &boost::gil::view(m_LeftIntegral)(0, 1)[0] - (char*) &boost::gil::view(m_LeftIntegral)(0, 0)[0]);
  cvInitImageHeader(&m_RightIntegralIpl, cvSize(m_RightIntegral.width(), m_RightIntegral.height()), IPL_DEPTH_32S, 1);
  cvSetData(&m_RightIntegralIpl, &boost::gil::view(m_RightIntegral)(0, 0)[0], (char*) &boost::gil::view(m_RightIntegral)(0, 1)[0] - (char*) &boost::gil::view(m_RightIntegral)(0, 0)[0]);

  cvInitImageHeader(&m_LeftSquaredIntegralIpl, cvSize(m_LeftSquaredIntegral.width(), m_LeftSquaredIntegral.height()), IPL_DEPTH_64F, 1);
  cvSetData(&m_LeftSquaredIntegralIpl, &boost::gil::view(m_LeftSquaredIntegral)(0, 0)[0], (char*) &boost::gil::view(m_LeftSquaredIntegral)(0, 1)[0] - (char*) &boost::gil::view(m_LeftSquaredIntegral)(0, 0)[0]);
  cvInitImageHeader(&m_RightSquaredIntegralIpl, cvSize(m_RightSquaredIntegral.width(), m_RightSquaredIntegral.height()), IPL_DEPTH_64F, 1);
  cvSetData(&m_RightSquaredIntegralIpl, &boost::gil::view(m_RightSquaredIntegral)(0, 0)[0], (char*) &boost::gil::view(m_RightSquaredIntegral)(0, 1)[0] - (char*) &boost::gil::view(m_RightSquaredIntegral)(0, 0)[0]);


  // some defaults that seem to work
  m_PropagationParams.N = 2;
  m_PropagationParams.Ct = 0.6f;
  m_PropagationParams.WinSizeX = 2;
  m_PropagationParams.WinSizeY = 2;
  m_PropagationParams.Dg = 1;
  m_PropagationParams.Tt = 200;
  m_PropagationParams.BorderX = 20;
  m_PropagationParams.BorderY = 20;
}


//-----------------------------------------------------------------------------
SequentialCpuQds::~SequentialCpuQds()
{
  // lucky us, all buffers are managed by gil or stl, so nothing to do
}


//-----------------------------------------------------------------------------
int SequentialCpuQds::GetWidth() const
{
  return m_Width;
}


//-----------------------------------------------------------------------------
int SequentialCpuQds::GetHeight() const
{
  return m_Height;
}


//-----------------------------------------------------------------------------
CvPoint SequentialCpuQds::GetMatch(int x, int y) const
{
  if ((x < 0) || (y < 0) || (x >= GetWidth()) || (y >= GetHeight()))
    throw std::runtime_error("Ref coordinate out of bounds");

  const boost::gil::dev2n16c_pixel_t& r = boost::gil::const_view(m_LeftRefMap)(x, y);
  return cvPoint(r[0], r[1]);
}


//-----------------------------------------------------------------------------
IplImage* SequentialCpuQds::CreateDisparityImage() const
{
  IplImage* dispimg = cvCreateImage(cvSize(GetWidth(), GetHeight()), IPL_DEPTH_8U, 4);

  // gil view that wraps the ipl image
  BOOST_AUTO(dst, boost::gil::interleaved_view(dispimg->width, dispimg->height, (boost::gil::rgba8_pixel_t*) dispimg->imageData, dispimg->widthStep));

  for (int y = 0; y < dispimg->height; ++y)
  {
    for (int x = 0; x < dispimg->width; ++x)
    {
      // two-channel pixel in the refmap. values point into the right image.
      const boost::gil::dev2n16c_pixel_t& r = boost::gil::const_view(m_LeftRefMap)(x, y);

      // output rgba pixel
      BOOST_AUTO(& p, dst(x, y));

      if (r[0] <= 0)
      {
        p[0] = 255;
        p[1] = 0;
        p[2] = 0;
        p[3] = 255;
        continue;
      }

      float   dx = x - r[0];
      float   dy = y - r[1];
      float   d = std::sqrt(dx*dx + dy*dy);
      d = std::max(d, 0.0f);
      d = std::min(d, (float) m_MaxDisparity);

      float   sd = /*255 -*/ 255.0f * (std::min(d - 0, (float) m_MaxDisparity) / ((float) m_MaxDisparity - 0));
      p[0] = (unsigned char) sd;
      p[1] = (unsigned char) sd;
      p[2] = (unsigned char) sd;
      p[3] = 255;
    }
  }

  return dispimg;
}


//-----------------------------------------------------------------------------
void SequentialCpuQds::InitSparseFeatures()
{
  CvSize  templateSize = cvSize((m_LeftIpl.width / 36) | 0x1, (m_LeftIpl.width / 36) | 0x1);
  double  minSeparationDistance = m_LeftIpl.width / 36;

  m_SparseFeaturesLeft.resize(NUM_MAX_FEATURES);

  // Detect features in the left image
  // note: opencv ignores eigenimage and tempimage!
  int   foundNumFeatures = m_SparseFeaturesLeft.size();
  cvGoodFeaturesToTrack(&m_LeftIpl, 0, 0, &m_SparseFeaturesLeft[0], &foundNumFeatures, 0.01, minSeparationDistance);
  m_SparseFeaturesLeft.resize(foundNumFeatures);

  // lets say we expect horizontal disparity
  //  and guess that on average we have 30 pixels.
  // so lets help the LKT, otherwise it sometimes finds lots of dodgy matches
  m_SparseFeaturesRight = m_SparseFeaturesLeft;
  for (unsigned int i = 0; i < m_SparseFeaturesRight.size(); ++i)
  {
    m_SparseFeaturesRight[i].x -= m_LeftIpl.width / 12;
  }

  m_FeatureStatus.resize(m_SparseFeaturesLeft.size());

  // match features from the current left frame to the current right frame
  cvCalcOpticalFlowPyrLK(&m_LeftIpl, &m_RightIpl, 0, 0,
      &m_SparseFeaturesLeft[0], &m_SparseFeaturesRight[0], foundNumFeatures,
      templateSize, 3, &m_FeatureStatus[0], 0,
      cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.0003), 0 | CV_LKFLOW_INITIAL_GUESSES);

#ifndef NDEBUG
  // help with debugging this
  // throw out all failed points, otherwise it's a mess trying to figure out which features were tracked and which were not
  for (int i = (int)m_SparseFeaturesLeft.size() - 1; i >= 0; --i)
  {
    if (m_FeatureStatus[i] == 0)
    {
      m_SparseFeaturesLeft.erase(m_SparseFeaturesLeft.begin() + i);
      m_SparseFeaturesRight.erase(m_SparseFeaturesRight.begin() + i);
      m_FeatureStatus.erase(m_FeatureStatus.begin() + i);
    }
  }
#endif
}


//-----------------------------------------------------------------------------
void SequentialCpuQds::QuasiDensePropagation()
{
  // we keep view objects around, for easier writing
  boost::gil::dev2n16_view_t      leftRef  = boost::gil::view(m_LeftRefMap);
  boost::gil::dev2n16_view_t      rightRef = boost::gil::view(m_RightRefMap);
  const boost::gil::gray8c_view_t leftTex  = boost::gil::const_view(m_LeftTexture);
  const boost::gil::gray8c_view_t rightTex = boost::gil::const_view(m_RightTexture);

  // Seed list
  std::priority_queue<Match, std::vector<Match>, std::less<Match> >   seeds;

  // Build a list of seeds from the starting features
  for (unsigned int i = 0; i < m_SparseFeaturesLeft.size(); i++)
  {
    if (m_FeatureStatus[i] != 0)
    {
      // Calculate correlation and store match in Seeds.
      Match m;
      // FIXME: check for negative coords! our guestimate in InitSparseFeatures() might have pushed these off
      m.p0 = RefPoint(m_SparseFeaturesLeft[i].x,  m_SparseFeaturesLeft[i].y);
      m.p1 = RefPoint(m_SparseFeaturesRight[i].x, m_SparseFeaturesRight[i].y);

      // Check if too close to boundary.
      if (!CheckBorder(m, m_PropagationParams.BorderX, m_PropagationParams.BorderY, m_LeftImg.width(), m_LeftImg.height()))
      {
        continue;
      }

      // Calculate the correlation threshold
      m.corr = Zncc_C1(m.p0.x, m.p0.y, m.p1.x, m.p1.y, m_PropagationParams.WinSizeX, 
                       boost::gil::const_view(m_LeftImg), boost::gil::const_view(m_RightImg),
                       boost::gil::const_view(m_LeftIntegral), boost::gil::const_view(m_RightIntegral),
                       boost::gil::const_view(m_LeftSquaredIntegral), boost::gil::const_view(m_RightSquaredIntegral));

      // Can we add it to the list
      if (m.corr > m_PropagationParams.Ct)
      {
        // FIXME: Check if this is unique (or assume it is due to prior supression)
        seeds.push(m);
        leftRef (m.p0.x, m.p0.y) = (boost::gil::dev2n16_pixel_t) m.p1;
        rightRef(m.p1.x, m.p1.y) = (boost::gil::dev2n16_pixel_t) m.p0;
      }
    }
  }

  // Do the propagation part
  while (!seeds.empty())
  {
    std::priority_queue<Match, std::vector<Match>, std::less<Match> >     localseeds;

    // Get the best seed at the moment
    Match m = seeds.top();
    seeds.pop();

    // Ignore the border
    if (!CheckBorder(m, m_PropagationParams.BorderX, m_PropagationParams.BorderY, m_LeftImg.width(), m_LeftImg.height()))
    {
      continue;
    }

    // For all neighbours in image 1
    for (int y = -m_PropagationParams.N; y <= m_PropagationParams.N; ++y)
    {
      for (int x = -m_PropagationParams.N; x <= m_PropagationParams.N; ++x)
      {
        RefPoint p0(m.p0.x + x, m.p0.y + y);

        // Check if its unique in ref
        if (leftRef(p0.x, p0.y)[0] != 0)
        {
          continue;
        }

        // Check the texture descriptor for a boundary
        if (leftTex(p0.x, p0.y) > m_PropagationParams.Tt) 
        {
          continue;
        }

        // For all candidate matches.
        for (int wy = -m_PropagationParams.Dg; wy <= m_PropagationParams.Dg; ++wy)
        {
          for (int wx = -m_PropagationParams.Dg; wx <= m_PropagationParams.Dg; ++wx)
          {
            RefPoint p1(m.p1.x + x + wx, m.p1.y + y + wy);

            // Check if its unique in ref
            if (rightRef(p1.x, p1.y)[0] != 0)
            {
              continue;
            }

            // Check the texture descriptor for a boundary
            if (rightTex(p1.x, p1.y) > m_PropagationParams.Tt) 
            {
              continue;
            }

            // Calculate ZNCC and store local match.
            float corr = Zncc_C1(p0.x, p0.y, p1.x, p1.y, m_PropagationParams.WinSizeX, 
                            boost::gil::const_view(m_LeftImg), boost::gil::const_view(m_RightImg),
                            boost::gil::const_view(m_LeftIntegral), boost::gil::const_view(m_RightIntegral),
                            boost::gil::const_view(m_LeftSquaredIntegral), boost::gil::const_view(m_RightSquaredIntegral));

            // push back if this is valid match
            if (corr > m_PropagationParams.Ct)
            {
              Match nm;
              nm.p0 = p0;
              nm.p1 = p1;
              nm.corr = corr;
              localseeds.push(nm);
            }
          }
        }
      }
    }

    // Get seeds from the local
    while (!localseeds.empty())
    {
      Match lm = localseeds.top();
      localseeds.pop();

      // Check if its unique in both ref and dst.
      if (leftRef(lm.p0.x, lm.p0.y)[0] != 0) 
      {
        continue;
      }
      if (rightRef(lm.p1.x, lm.p1.y)[0] != 0) 
      {
        continue;
      }

      // Unique match
      leftRef (lm.p0.x, lm.p0.y) = (boost::gil::dev2n16_pixel_t) lm.p1;
      rightRef(lm.p1.x, lm.p1.y) = (boost::gil::dev2n16_pixel_t) lm.p0;

      // Add to the seed list
      seeds.push(lm);
    }
  } // while (global) seed list is not empty
}


//-----------------------------------------------------------------------------
void SequentialCpuQds::Process(const IplImage* left, const IplImage* right)
{
  assert(left  != 0);
  assert(right != 0);

  assert(left->nChannels == right->nChannels);
  assert(left->width  == right->width);
  assert(left->height == right->height);

  if ((left->width  != GetWidth()) ||
      (left->height != GetHeight()))
  {
    throw std::runtime_error("Image size does not match");
  }

  // the opencv channel layout is BGR by default
  // but lucky us, the greyscale conversion is not weighted
  // so it doesnt matter whether we say RGB or BGR.
  switch (left->nChannels)
  {
    case 3:
      cvCvtColor(left,  &m_LeftIpl,  CV_RGB2GRAY);
      cvCvtColor(right, &m_RightIpl, CV_RGB2GRAY);
      break;
    case 4:
      cvCvtColor(left,  &m_LeftIpl,  CV_RGBA2GRAY);
      cvCvtColor(right, &m_RightIpl, CV_RGBA2GRAY);
      break;

    default:
      throw std::runtime_error("Unknown image format");
  }


  // run an initial sparse feature matching step
  InitSparseFeatures();

  // build texture homogeneity reference maps.
  BuildTextureDescriptor(boost::gil::const_view(m_LeftImg),  boost::gil::view(m_LeftTexture));
  BuildTextureDescriptor(boost::gil::const_view(m_RightImg), boost::gil::view(m_RightTexture));

  // generate the integral images for fast variable window correlation calculations
  cvIntegral(&m_LeftIpl,  &m_LeftIntegralIpl,  &m_LeftSquaredIntegralIpl);
  cvIntegral(&m_RightIpl, &m_RightIntegralIpl, &m_RightSquaredIntegralIpl);

  // we need to reset these, so that we get a fresh stereo matching
  boost::gil::fill_pixels(boost::gil::view(m_LeftRefMap),  boost::gil::dev2n16_pixel_t(0, 0));
  boost::gil::fill_pixels(boost::gil::view(m_RightRefMap), boost::gil::dev2n16_pixel_t(0, 0));

  QuasiDensePropagation();
}


} // namespace

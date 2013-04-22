/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "SequentialCpuQds.h"
#include <queue>
#include <boost/gil/gil_all.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/video/tracking.hpp>

#ifdef _OMP
#include <omp.h>
#endif


namespace mitk 
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
// NOTE: returns values in the range of [-1...+1]
// FIXME: this needs proper unit-testing! desperately!
static float zncc_c1(int p0x, int p0y, int p1x, int p1y, int w, boost::gil::gray8c_view_t img0, boost::gil::gray8c_view_t img1, boost::gil::gray32sc_view_t integral0, boost::gil::gray32sc_view_t integral1, boost::gil::gray64fc_view_t square0, boost::gil::gray64fc_view_t square1)
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

  // FIXME: is this supposed to happen??
  //        i suddenly get all sorts of weird errors because of this, never seen this before
  if (s0 <= 0)
    return 0;
  if (s1 <= 0)
    return 0;

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
static void BuildTextureDescriptor(const boost::gil::gray8c_view_t src, const boost::gil::gray8_view_t dst)
{
  assert(src.dimensions() == dst.dimensions());

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
void SequentialCpuQds::InitSparseFeatures()
{
  CvSize  templateSize = cvSize((m_LeftIpl.width / 72) | 0x1, (m_LeftIpl.width / 72) | 0x1);
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
  for (int i = 0; i < m_SparseFeaturesRight.size(); ++i)
  {
    m_SparseFeaturesRight[i].x -= m_LeftIpl.width / 12;
  }

  m_FeatureStatus.resize(m_SparseFeaturesLeft.size());

  // match features from the current left frame to the current right frame
  cvCalcOpticalFlowPyrLK(&m_LeftIpl, &m_RightIpl, 0, 0,
      &m_SparseFeaturesLeft[0], &m_SparseFeaturesRight[0], foundNumFeatures,
      templateSize, 3, &m_FeatureStatus[0], 0,
      cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.0003), CV_LKFLOW_INITIAL_GUESSES);
}


//-----------------------------------------------------------------------------
void SequentialCpuQds::QuasiDensePropagation()
{
#if 0
	int x,y,i,wx,wy,off;


	// for fast processing initialize some pointers to the data 
	data0 = (unsigned char *)imgL->imageData;
	data1 = (unsigned char *)imgR->imageData;
	sum0  = (__int32*)mIntegralImage[0]->imageData;
	sum1  = (__int32*)mIntegralImage[1]->imageData;
	ssum0 = (double*)mIntegralImageSq[0]->imageData;
	ssum1 = (double*)mIntegralImageSq[1]->imageData;

	Step = imgL->widthStep;
	Steps = mIntegralImage[0]->width;

#endif

  // we keep view objects around, for easier writing
  boost::gil::dev2n16_view_t      leftRef  = boost::gil::view(m_LeftRefMap);
  boost::gil::dev2n16_view_t      rightRef = boost::gil::view(m_RightRefMap);
  const boost::gil::gray8c_view_t leftTex  = boost::gil::const_view(m_LeftTexture);
  const boost::gil::gray8c_view_t rightTex = boost::gil::const_view(m_RightTexture);

  // Seed list
  std::priority_queue<Match, std::vector<Match>, std::less<Match> >   seeds;

  // Build a list of seeds from the starting features
  for (int i = 0; i < m_SparseFeaturesLeft.size(); i++)
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
      m.corr = zncc_c1(m.p0.x, m.p0.y, m.p1.x, m.p1.y, m_PropagationParams.WinSizeX, 
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
        for (int wy= -m_PropagationParams.Dg; wy <= m_PropagationParams.Dg; ++wy)
        {
          for (int wx= -m_PropagationParams.Dg; wx <= m_PropagationParams.Dg; ++wx)
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
            float corr = zncc_c1(p0.x, p0.y, p1.x, p1.y, m_PropagationParams.WinSizeX, 
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

  // generate the intergal images for fast variable window correlation calculations
  cvIntegral(&m_LeftIpl,  &m_LeftIntegralIpl,  &m_LeftSquaredIntegralIpl);
  cvIntegral(&m_RightIpl, &m_RightIntegralIpl, &m_RightSquaredIntegralIpl);

  // we need to reset these, so that we get a fresh stereo matching
  boost::gil::fill_pixels(boost::gil::view(m_LeftRefMap),  boost::gil::dev2n16_pixel_t(0, 0));
  boost::gil::fill_pixels(boost::gil::view(m_RightRefMap), boost::gil::dev2n16_pixel_t(0, 0));

  QuasiDensePropagation();
}


} // namespace

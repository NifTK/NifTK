/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#if defined(_MSC_VER)
//#pragma warning ( disable : 4786 )
#endif


#include <iostream>
#include <cstdlib>
#include <cassert>
#include <string>
#include <stdexcept>
#include <CameraCalibration/niftkUndistortionKernel.h>
#include <boost/gil/gil_all.hpp>
#include <cuda_runtime_api.h>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui_c.h>


std::pair<int, int> DecodePixelToCoordinate(boost::gil::rgba8_pixel_t p)
{
  // alpha is not a coordinate
  return std::make_pair(p[0] + (p[1] << 8), p[2]);
}

static boost::gil::rgba8_image_t CreateTestImage(int width, int height)
{
  boost::gil::rgba8_image_t   img(width, height);
  boost::gil::rgba8_view_t    view = boost::gil::view(img);

  for (int y = 0; y < view.height(); ++y)
  {
    for (int x = 0; x < view.width(); ++x)
    {
      unsigned char     r = x & 0xFF;
      unsigned char     g = (x >> 8) & 0xFF;
      unsigned char     b = y & 0xFF;
      unsigned char     a = 255;//(y >> 8) & 0xFF;
      view(x, y) = boost::gil::rgba8_pixel_t(r, g, b, a);
    }
  }

  return img;
}


static boost::gil::rgba8_image_t RunKernel(boost::gil::rgba8_view_t input, const cv::Mat& cammat, const cv::Mat& distmat)
{
  cudaError_t   err = cudaSuccess;

  std::size_t   bufsize = input.width() * input.height() * sizeof(boost::gil::rgba8_pixel_t);
  std::size_t   imgsize = (char*) &input(input.width() - 1, input.height() - 1)[3] - (char*) &input(0, 0)[0] + 1;
  // sanity check
  assert(bufsize == imgsize);

  // result buffer
  void*   outptr = 0;
  err = cudaMalloc(&outptr, bufsize);
  if (err == cudaErrorMemoryAllocation)
  {
    throw std::bad_alloc();
  }
  err = cudaMemset(outptr, 0xFF, bufsize);
  assert(err == cudaSuccess);

  // input buffer
  void*   inptr = 0;
  err = cudaMalloc(&inptr, bufsize);
  if (err == cudaErrorMemoryAllocation)
  {
    throw std::bad_alloc();
  }

  err = cudaMemcpy(inptr, &input(0, 0)[0], imgsize, cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);

  cudaResourceDesc    resdesc = {cudaResourceTypePitch2D, 0};
  resdesc.res.pitch2D.devPtr = inptr;
  resdesc.res.pitch2D.desc.x = resdesc.res.pitch2D.desc.y = resdesc.res.pitch2D.desc.z = resdesc.res.pitch2D.desc.w = 8;
  resdesc.res.pitch2D.desc.f = cudaChannelFormatKindUnsigned;
  resdesc.res.pitch2D.width  = input.width();
  resdesc.res.pitch2D.height = input.height();
  resdesc.res.pitch2D.pitchInBytes = (char*) &input(0, 1)[0] - (char*) &input(0, 0)[0];

  cudaTextureDesc     texdesc = {cudaAddressModeWrap};
  texdesc.addressMode[0] = texdesc.addressMode[1] = texdesc.addressMode[2] = cudaAddressModeBorder;//cudaAddressModeClamp;
  texdesc.filterMode = cudaFilterModeLinear;
  texdesc.readMode = cudaReadModeNormalizedFloat;
  texdesc.normalizedCoords = 1;

  cudaTextureObject_t   texobj;
  err = cudaCreateTextureObject(&texobj, &resdesc, &texdesc, 0);
  assert(err == cudaSuccess);


  niftk::RunUndistortionKernel((char*) outptr, input.width(), input.height(), texobj, (float*) cammat.data, (float*) distmat.data, 0);

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
  {
    throw std::runtime_error("cuda kernel failed");
  }


  boost::gil::rgba8_image_t   resultimg(input.width(), input.height());
  err = cudaMemcpy(&boost::gil::view(resultimg)(0, 0)[0], outptr, imgsize, cudaMemcpyDeviceToHost);
  assert(err == cudaSuccess);

  // cleanup
  err = cudaFree(outptr);
  assert(err == cudaSuccess);
  err = cudaFree(inptr);
  assert(err == cudaSuccess);
  err = cudaDestroyTextureObject(texobj);
  assert(err == cudaSuccess);

  return resultimg;
}


static bool CompareAgainstOpenCV(boost::gil::rgba8_view_t input, const cv::Mat& cammat, const cv::Mat& distmat, boost::gil::rgba8_view_t kernelresult, const char* testname = "test")
{
  IplImage* remapx = cvCreateImage(cvSize(input.width(), input.height()), IPL_DEPTH_32F, 1);
  IplImage* remapy = cvCreateImage(cvSize(input.width(), input.height()), IPL_DEPTH_32F, 1);

  // the old-style CvMat will reference the memory in the new-style cv::Mat.
  // that's why we keep these in separate variables.
  CvMat cam  = cammat;
  CvMat dist = distmat;
  cvInitUndistortMap(&cam, &dist, remapx, remapy);


  IplImage  inputipl;
  cvInitImageHeader(&inputipl, cvSize(input.width(), input.height()), IPL_DEPTH_8U, 4);
  cvSetData(&inputipl, &input(0, 0)[0], (char*) &input(0, 1)[0] - (char*) &input(0, 0)[0]);

  boost::gil::rgba8_image_t    cvoutput(input.width(), input.height());
  IplImage  outputipl;
  cvInitImageHeader(&outputipl, cvSize(cvoutput.width(), cvoutput.height()), IPL_DEPTH_8U, 4);
  cvSetData(&outputipl, &boost::gil::view(cvoutput)(0, 0)[0], (char*) &boost::gil::view(cvoutput)(0, 1)[0] - (char*) &boost::gil::view(cvoutput)(0, 0)[0]);

  cvRemap(&inputipl, &outputipl, remapx, remapy, CV_INTER_LINEAR, cvScalarAll(0));

  cvReleaseImage(&remapx);
  cvReleaseImage(&remapy);


  IplImage  kernelipl;
  cvInitImageHeader(&kernelipl, cvSize(kernelresult.width(), kernelresult.height()), IPL_DEPTH_8U, 4);
  cvSetData(&kernelipl, &kernelresult(0, 0)[0], (char*) &kernelresult(0, 1)[0] - (char*) &kernelresult(0, 0)[0]);

  cvSaveImage((std::string(testname) + "-kernelresult.png").c_str(), &kernelipl);
  cvSaveImage((std::string(testname) + "-opencvresult.png").c_str(), &outputipl);

  // FIXME: erode alpha channel of kernel result
  {
    CvArr*      in[] = {&kernelipl};
    IplImage*   alphaonly = cvCreateImage(cvSize(kernelipl.width, kernelipl.height), kernelipl.depth, 1);
    CvArr*      out[] = {alphaonly};
    int         pairs[] = {3, 0, 0, 3};
    cvMixChannels((const CvArr**) &in[0], 1, &out[0], 1, &pairs[0], 1);

    // supports in-place
    cvErode(alphaonly, alphaonly, 0, 1);

    cvMixChannels((const CvArr**) &out[0], 1, &in[0], 1, &pairs[2], 1);

    cvReleaseImage(&alphaonly);
  }

  cvSaveImage((std::string(testname) + "-kernelresult-after.png").c_str(), &kernelipl);

  int   maxpixeldifference = 0;
  int   numpixelsdifferent = 0;
  // note: opencv messes up around the image border
  for (int y = 1; y < cvoutput.height() - 1; ++y)
  {
    for (int x = 1; x < cvoutput.width() - 1; ++x)
    {
      boost::gil::rgba8_pixel_t   opencvpixel = boost::gil::view(cvoutput)(x, y);
      boost::gil::rgba8_pixel_t   kernelpixel = kernelresult(x, y);

      std::pair<int, int>   opencvcoord = DecodePixelToCoordinate(opencvpixel);
      std::pair<int, int>   kernelcoord = DecodePixelToCoordinate(kernelpixel);

      // opencv will fill in some garbage in areas that do not map back to input.
      // the kernel will just fill these pixels with alpha zero.
      // so ignore pixels that do not have fully-opaque alpha.
      if (kernelpixel[3] >= 255)
      {
        int   pixeldiff = std::max(std::abs((int) opencvpixel[0] - (int) kernelpixel[0]),
                          std::max(std::abs((int) opencvpixel[1] - (int) kernelpixel[1]),
                          std::max(std::abs((int) opencvpixel[2] - (int) kernelpixel[2]),
                                   std::abs((int) opencvpixel[3] - (int) kernelpixel[3]))));
        int   coorddiff = std::max(std::abs((int) opencvcoord.first  - (int) kernelcoord.first),
                                   std::abs((int) opencvcoord.second - (int) kernelcoord.second));

        if (pixeldiff > 5)
        {
          std::cerr << testname << ": Pixel at " << x << ',' << y << " differs too much" << std::endl;

          if (coorddiff > 5)
            ++numpixelsdifferent;
        }

        maxpixeldifference = std::max(maxpixeldifference, std::min(pixeldiff, coorddiff));
      }
    }
  }


  // current testing puts the max difference at 5
  if (maxpixeldifference > 5)
  {
    std::cerr << testname <<": FAIL: Difference between any two pixels is too large: " << maxpixeldifference << std::endl;
    return false;
  }
  if (numpixelsdifferent > std::max(input.width(), input.height()))
  {
    std::cerr << testname << ": FAIL: Too many different pixels: " << numpixelsdifferent << std::endl;
    return false;
  }

  return true;
}


static bool IdentityUndistortion()
{
  boost::gil::rgba8_image_t   testimg = CreateTestImage(1024, 1024);

  float   cammatdata[9] =
  {
    testimg.width(), 0, testimg.width() / 2,
    0, testimg.height(), testimg.height() / 2,
    0, 0, 1
  };
  cv::Mat   cammat(3, 3, CV_32F, &cammatdata[0]);

  float   distmatdata[4] = {0, 0, 0, 0};
  cv::Mat   distmat(1, 4, CV_32F, &distmatdata[0]);

  boost::gil::rgba8_image_t kernelresult = RunKernel(boost::gil::view(testimg), cammat, distmat);

  bool  match = CompareAgainstOpenCV(boost::gil::view(testimg), cammat, distmat, boost::gil::view(kernelresult), "identity");
  if (match)
    std::cerr << "SUCCESS Identity undistortion" << std::endl;
  return match;
}


static bool RadialUndistortion()
{
  boost::gil::rgba8_image_t   testimg = CreateTestImage(1024, 1024);

  float   cammatdata[9] =
  {
    testimg.width(), 0, testimg.width() / 2,
    0, testimg.height(), testimg.height() / 2,
    0, 0, 1
  };
  cv::Mat   cammat(3, 3, CV_32F, &cammatdata[0]);

  float   distmatdata[4] = {0.5, 0, 0, 0};
  cv::Mat   distmat(1, 4, CV_32F, &distmatdata[0]);

  boost::gil::rgba8_image_t kernelresult = RunKernel(boost::gil::view(testimg), cammat, distmat);

  bool  match = CompareAgainstOpenCV(boost::gil::view(testimg), cammat, distmat, boost::gil::view(kernelresult), "radialx");
  if (match)
    std::cerr << "SUCCESS Radial undistortion" << std::endl;
  return match;
}


static bool TangentialUndistortion()
{
  boost::gil::rgba8_image_t   testimg = CreateTestImage(1024, 1024);

  float   cammatdata[9] =
  {
    testimg.width(), 0, testimg.width() / 2,
    0, testimg.height(), testimg.height() / 2,
    0, 0, 1
  };
  cv::Mat   cammat(3, 3, CV_32F, &cammatdata[0]);

  float   distmatdata[4] = {0, 0, 0.2, 0.2};
  cv::Mat   distmat(1, 4, CV_32F, &distmatdata[0]);

  boost::gil::rgba8_image_t kernelresult = RunKernel(boost::gil::view(testimg), cammat, distmat);

  bool  match = CompareAgainstOpenCV(boost::gil::view(testimg), cammat, distmat, boost::gil::view(kernelresult), "tangential");
  if (match)
    std::cerr << "SUCCESS tangential undistortion" << std::endl;
  return match;
}


static bool SomeOfEverythingUndistortion()
{
  boost::gil::rgba8_image_t   testimg = CreateTestImage(1024, 1024);

  float   cammatdata[9] =
  {
    testimg.width(), 0, testimg.width() / 2,
    0, testimg.height(), testimg.height() / 2,
    0, 0, 1
  };
  cv::Mat   cammat(3, 3, CV_32F, &cammatdata[0]);

  float   distmatdata[4] = {0.5, -0.4, -0.5, 0.4};
  cv::Mat   distmat(1, 4, CV_32F, &distmatdata[0]);

  boost::gil::rgba8_image_t kernelresult = RunKernel(boost::gil::view(testimg), cammat, distmat);

  bool  match = CompareAgainstOpenCV(boost::gil::view(testimg), cammat, distmat, boost::gil::view(kernelresult), "someofeverything");
  if (match)
    std::cerr << "SUCCESS someofeverything undistortion" << std::endl;
  return match;
}


static bool OffCenterUndistortion()
{
  boost::gil::rgba8_image_t   testimg = CreateTestImage(1024, 1024);

  float   cammatdata[9] =
  {
    testimg.width(), 0, testimg.width() / 3,
    0, testimg.height(), testimg.height() / 3,
    0, 0, 1
  };
  cv::Mat   cammat(3, 3, CV_32F, &cammatdata[0]);

  float   distmatdata[4] = {0.1, 0.1, 0.1, 0.1};
  cv::Mat   distmat(1, 4, CV_32F, &distmatdata[0]);

  boost::gil::rgba8_image_t kernelresult = RunKernel(boost::gil::view(testimg), cammat, distmat);

  bool  match = CompareAgainstOpenCV(boost::gil::view(testimg), cammat, distmat, boost::gil::view(kernelresult), "offcenter");
  if (match)
    std::cerr << "SUCCESS offcenter undistortion" << std::endl;
  return match;
}


int niftkCUDAKernelsUndistortionTest(int argc, char* argv[])
{
  bool    result = true;

  try
  {
    result &= IdentityUndistortion();
    result &= RadialUndistortion();
    result &= TangentialUndistortion();
    result &= SomeOfEverythingUndistortion();
    result &= OffCenterUndistortion();
  }
  catch (const std::exception& e)
  {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return result ? EXIT_SUCCESS : EXIT_FAILURE;
}

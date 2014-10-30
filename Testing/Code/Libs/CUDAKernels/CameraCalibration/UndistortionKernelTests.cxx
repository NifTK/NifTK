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
#include <CameraCalibration/UndistortionKernel.h>
#include <boost/gil/gil_all.hpp>
#include <cuda_runtime_api.h>


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
      unsigned char     a = (y >> 8) & 0xFF;
      view(x, y) = boost::gil::rgba8_pixel_t(r, g, b, a);
    }
  }

  return img;
}


static bool IdentityUndistortion()
{
  boost::gil::rgba8_image_t   testimg = CreateTestImage(1024, 1024);


  cudaError_t   err = cudaSuccess;

  std::size_t   bufsize = testimg.width() * testimg.height() * sizeof(boost::gil::rgba8_pixel_t);
  std::size_t   imgsize = (char*) &boost::gil::view(testimg)(testimg.width() - 1, testimg.height() - 1)[3] - (char*) &boost::gil::view(testimg)(0, 0)[0] + 1;
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

  err = cudaMemcpy(inptr, &boost::gil::view(testimg)(0, 0)[0], imgsize, cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);

  cudaResourceDesc    resdesc = {cudaResourceTypePitch2D, 0};
  resdesc.res.pitch2D.devPtr = inptr;
  resdesc.res.pitch2D.desc.x = resdesc.res.pitch2D.desc.y = resdesc.res.pitch2D.desc.z = resdesc.res.pitch2D.desc.w = 8;
  resdesc.res.pitch2D.desc.f = cudaChannelFormatKindUnsigned;
  resdesc.res.pitch2D.width  = testimg.width();
  resdesc.res.pitch2D.height = testimg.height();
  resdesc.res.pitch2D.pitchInBytes = (char*) &boost::gil::view(testimg)(0, 1)[0] - (char*) &boost::gil::view(testimg)(0, 0)[0];

  cudaTextureDesc     texdesc = {cudaAddressModeWrap};
  texdesc.addressMode[0] = texdesc.addressMode[1] = texdesc.addressMode[2] = cudaAddressModeClamp;
  texdesc.filterMode = cudaFilterModeLinear;
  texdesc.readMode = cudaReadModeNormalizedFloat;
  texdesc.normalizedCoords = 1;

  cudaTextureObject_t   texobj;
  err = cudaCreateTextureObject(&texobj, &resdesc, &texdesc, 0);
  assert(err == cudaSuccess);

  float   cammat[9] = 
  {
    testimg.width(), 0, testimg.width() / 2,
    0, testimg.height(), testimg.height() / 2,
    0, 0, 1
  };
  float   distmat[4] = {0, 0, 0, 0};

  RunUndistortionKernel((char*) outptr, testimg.width(), testimg.height(), texobj, &cammat[0], &distmat[0], 0);

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess)
  {
    throw std::runtime_error("cuda kernel failed");
  }


  boost::gil::rgba8_image_t   resultimg(testimg);
  err = cudaMemcpy(&boost::gil::view(resultimg)(0, 0)[0], outptr, imgsize, cudaMemcpyDeviceToHost);
  assert(err == cudaSuccess);



  // cleanup
  err = cudaFree(outptr);
  assert(err == cudaSuccess);
  err = cudaFree(inptr);
  assert(err == cudaSuccess);
  err = cudaDestroyTextureObject(texobj);
  assert(err == cudaSuccess);

  // we made it till here, so all is good.
  return true;
}


int niftkCUDAKernelsUndistortionTest(int argc, char* argv[])
{
  bool    result = true;

  result &= IdentityUndistortion();

  return result ? EXIT_SUCCESS : EXIT_FAILURE;
}

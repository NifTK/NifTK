/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/

#include "niftkundistortionLauncher.h"
#include <niftkCUDAManager.h>
#include <niftkLightweightCUDAImage.h>
#include <CameraCalibration/niftkUndistortionKernel.h>
#include <cassert>

namespace niftk
{

//-----------------------------------------------------------------------------
void UndistortionLauncher(char *inputImageData,
                          int width,
                          int height,
                          int widthStep,
                          float *intrinsics,
                          float *distortion,
                          char *outputImageData)
{
  CUDAManager*    cm = CUDAManager::GetInstance();
  cudaStream_t    stream = cm->GetStream("undistortion");

  WriteAccessor inputWA = cm->RequestOutputImage(width, height, 4);

  // source of memcpy is not page-locked, so would never be async.
  cudaError_t err = cudaMemcpyAsync(inputWA.m_DevicePointer, inputImageData, widthStep * height,
                                    cudaMemcpyHostToDevice, stream);
  assert(err == cudaSuccess);

  // we need another image, this time for actual output.
  WriteAccessor outputWA = cm->RequestOutputImage(width, height, 4);

  // wrap the linear memory with our input data in a texture object,
  // so that we can read filtered results.
  cudaResourceDesc resdesc = {cudaResourceTypePitch2D, 0};
  resdesc.res.pitch2D.devPtr = inputWA.m_DevicePointer;
  resdesc.res.pitch2D.desc.x = 8;
  resdesc.res.pitch2D.desc.y = 8;
  resdesc.res.pitch2D.desc.z = 8;
  resdesc.res.pitch2D.desc.w = 8;
  resdesc.res.pitch2D.desc.f = cudaChannelFormatKindUnsigned;
  resdesc.res.pitch2D.width  = width;
  resdesc.res.pitch2D.height = height;
  resdesc.res.pitch2D.pitchInBytes = widthStep;

  cudaTextureDesc texdesc = {cudaAddressModeWrap};
  texdesc.addressMode[0] = texdesc.addressMode[1] = texdesc.addressMode[2] = cudaAddressModeClamp;
  texdesc.filterMode = cudaFilterModeLinear;
  texdesc.normalizedCoords = 1;
  // could be cudaReadModeNormalizedFloat to have automatic conversion to floating point.
  texdesc.readMode = cudaReadModeNormalizedFloat;//cudaReadModeElementType;

  cudaTextureObject_t texobj;
  err = cudaCreateTextureObject(&texobj, &resdesc, &texdesc, 0);
  assert(err == cudaSuccess);

  niftk::RunUndistortionKernel((char*) outputWA.m_DevicePointer, width, height, texobj, intrinsics, distortion, stream);

  // even though we don't need the image that holds input data,
  // we still have to finalise it. but we can simply discard the output LightweightCUDAImage.
  cm->Finalise(inputWA, stream);

  // this time, we do want to keep the result
  LightweightCUDAImage outputLWCI = cm->Finalise(outputWA, stream);

  // the interface for CUDAManager does intentionally not have readback functionality.
  // so we need to request read access to our own output.
  // (obviously we could shortcut this, we have the device pointers. but this exercise is about api.)
  ReadAccessor outputRA = cm->RequestReadAccess(outputLWCI);

  // even though the name has async in it, it is synchronous because the destination is not page-locked.
  // but we want the extra stream parameter so that it syncs onto our compute stream.
  err = cudaMemcpyAsync(outputImageData, outputRA.m_DevicePointer, widthStep * height, cudaMemcpyDeviceToHost, stream);
  assert(err == cudaSuccess);

  err = cudaDestroyTextureObject(texobj);
  assert(err == cudaSuccess);

  cm->Autorelease(outputRA, stream);
}

} // end namespace

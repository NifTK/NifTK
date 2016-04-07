/*=============================================================================

  NifTK: A software platform for medical image computing.

  Copyright (c) University College London (UCL). All rights reserved.

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.

  See LICENSE.txt in the top level directory for details.

=============================================================================*/
#ifndef niftkUndistortionLauncher_h
#define niftkUndistortionLauncher_h

#include "niftkCUDAExports.h"

namespace niftk
{

/**
* \brief Runs the CUDA based image undistortion.
* \see niftk::RunUndistortionKernel
* \param hostInputImageData host-side input 4-channel image data, such as might be obtained from an IplImage
* \param width the width of the image in pixels
* \param height the height of the image in pixels
* \param widthStep = width * channels (4)
* \param intrinsics array of 9 floats containing camera intrinsics
* \param distortion array of 4 floats containing distortion coefficients
*/
void NIFTKCUDA_EXPORT UndistortionLauncher(char *hostInputImageData,
                                           int width,
                                           int height,
                                           int widthStep,
                                           float *intrinsics,
                                           float *distortion,
                                           char *hostOutputImageData);

} // end namespace

#endif

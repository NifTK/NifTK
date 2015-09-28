#/*============================================================================
#
#  NifTK: A software platform for medical image computing.
#
#  Copyright (c) University College London (UCL). All rights reserved.
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.
#
#  See LICENSE.txt in the top level directory for details.
#
#============================================================================*/

set(_depends MitkOpenCVVideoSupport MitkCameraCalibration niftkCore niftkOpenCVUtils niftkcommon)

# if cuda is available and enabled then we have possibility of doing
# some computer vision stuff on the gpu.
if(CUDA_FOUND AND NIFTK_USE_CUDA)
  list(APPEND _depends niftkCUDA)
endif()

MITK_CREATE_MODULE(
  INCLUDE_DIRS CameraCalibration VideoProcessing Registration demo VideoTrackerMatching UltrasoundCalibration PivotCalibration Features
  DEPENDS ${_depends}
  PACKAGE_DEPENDS OpenCV Boost|filesystem
)

if(CUDA_FOUND AND NIFTK_USE_CUDA)
  if(MSVC AND NIFTK_DELAYLOAD_CUDA)
    # all of the nvidia-driver/cuda dependencies should be delay-loaded
    #  so this library would at least load on a machine that doesnt have nv hardware.
    set_target_properties(niftkOpenCV PROPERTIES LINK_FLAGS 
      "/DELAYLOAD:nvcuda.dll /DELAYLOAD:${CUDA_CUDART_DLL_NAME}")
  endif()
endif()

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkOpenCV)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()

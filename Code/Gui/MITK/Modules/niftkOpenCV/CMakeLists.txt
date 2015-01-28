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


set(_depends MitkOpenCVVideoSupport MitkCameraCalibration niftkCore)

# if cuda is available and enabled then we have possibility of doing
# some computer vision stuff on the gpu.
if(CUDA_FOUND AND NIFTK_USE_CUDA)
  list(APPEND _depends niftkCUDA)
endif()

MITK_CREATE_MODULE(niftkOpenCV
  EXPORT_DEFINE NIFTKOPENCV_EXPORT
  INCLUDE_DIRS Common CameraCalibration VideoProcessing Registration demo VideoTrackerMatching UltrasoundCalibration PivotCalibration Features
  DEPENDS ${_depends}
  PACKAGE_DEPENDS OpenCV
)

target_link_libraries(niftkOpenCV
  niftkcommon
  )


if(CUDA_FOUND AND NIFTK_USE_CUDA)
  # the mitk module niftkOpenCV will not contain any cuda kernels but it
  # can still call api functions, etc.
  
  # FIXME: this dependency on niftkCUDAKernels should probably go to niftkCUDA, which wraps
  #        up all kernel calls nicely for other modules to call.
  include_directories(${CUDA_TOOLKIT_INCLUDE} ${CMAKE_SOURCE_DIR}/Code/Libs/CUDAKernels)
  target_link_libraries(niftkOpenCV ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} niftkCUDAKernels)
  
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

add_subdirectory(Testing)

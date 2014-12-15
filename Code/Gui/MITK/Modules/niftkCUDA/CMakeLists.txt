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

MITK_CREATE_MODULE(niftkCUDA
  EXPORT_DEFINE NIFTKCUDA_EXPORT
  INCLUDE_DIRS CUDAManager CUDAImage VLInterface
  DEPENDS MitkCore
  PACKAGE_DEPENDS Qt4|QtCore Qt4|QtOpenGL
)

include_directories(
  ${CUDA_TOOLKIT_INCLUDE}
  ${CMAKE_SOURCE_DIR}/Code/Libs/CUDAKernels
)

target_link_libraries(niftkCUDA
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  niftkCUDAKernels
)


if(MSVC AND NIFTK_DELAYLOAD_CUDA)
  # all of the nvidia-driver/cuda dependencies should be delay-loaded
  #  so this library would at least load on a machine that doesnt have nv hardware
  set_target_properties(niftkCUDA PROPERTIES LINK_FLAGS 
    "/DELAYLOAD:nvcuda.dll /DELAYLOAD:${CUDA_CUDART_DLL_NAME}")
endif()

if (NIFTK_USE_COTIRE AND COMMAND cotire)
  cotire(niftkCUDA)
endif()

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()

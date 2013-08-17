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

find_package(CUDA REQUIRED)

MITK_CREATE_MODULE(niftkNVidiaGui
  EXPORT_DEFINE NIFTKNVIDIAGUI_EXPORT
  INCLUDE_DIRS DataSources
  DEPENDS niftkIGIGui niftkNVidia
  PACKAGE_DEPENDS QT
  QT_MODULE 
)

include_directories(${CMAKE_SOURCE_DIR}/Code/Libs/NVAPI/video)
include_directories(${QT_QTOPENGL_INCLUDE_DIR})
include_directories(${CUDA_TOOLKIT_INCLUDE})

# stuffing this into DEPENDS above doesnt work for some reason
target_link_libraries(niftkNVidiaGui
  niftknvapi
  mitkOpenCVVideoSupport
  Winmm.lib
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  )


if(MSVC)
  # all of the nvidia-driver/cuda dependencies should be delay-loaded
  #  so this library would at least load on a machine that doesnt have nv hardware
  set_target_properties(niftkNVidiaGui PROPERTIES LINK_FLAGS 
    "/DELAYLOAD:nvcuda.dll /DELAYLOAD:${CUDA_CUDART_DLL_NAME}")
endif(MSVC)


if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()

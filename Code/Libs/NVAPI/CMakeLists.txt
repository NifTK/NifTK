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
 
set(niftknvapi_SRCS
  niftkNVAPICommon.cxx
  glew/glew.c
  video/compress.cxx
  video/device.cxx
  video/deviceimpl.cxx
  video/frame.cxx
  video/sdiinput.cxx
  video/sdioutput.cxx
  video/rgba2nv12.cu
)

# with all the itk etc stuff, nvcc dies with "command line too long".
# so instead drop off all the not needed stuff and start clean.
# http://www.cmake.org/Bug/print_bug_page.php?bug_id=12198
set_directory_properties( PROPERTIES INCLUDE_DIRECTORIES "" )

include_directories(${CMAKE_SOURCE_DIR}/Code/Libs/NVAPI/video)
include_directories(${CMAKE_SOURCE_DIR}/Code/Libs/NVAPI/glew)
include_directories(${NVAPI_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CUDA_TOOLKIT_INCLUDE})

# add_library(niftknvapi ${niftknvapi_SRCS})
CUDA_ADD_LIBRARY(niftknvapi ${niftknvapi_SRCS})

if(BUILD_SHARED_LIBS)
  if(WIN32)
    add_definitions(-DNIFTKNVAPI_WINDOWS_EXPORT)
    add_definitions(-DLIBVIDEO_BUILDING_DLL)
  endif(WIN32)
endif(BUILD_SHARED_LIBS)


get_filename_component(cudaencoderpath ${CUDA_CUDART_LIBRARY} PATH)

target_link_libraries(niftknvapi  
  ${NVAPI_LIBRARY}
  ${OPENGL_gl_LIBRARY}
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${cudaencoderpath}/nvcuvenc.lib
  ${cudaencoderpath}/nvcuvid.lib
)

if(MSVC)
  # all of the nvidia-driver/cuda dependencies should be delay-loaded
  #  so this library would at least load on a machine that doesnt have nv hardware
  set_target_properties(niftknvapi PROPERTIES LINK_FLAGS 
    "/DELAYLOAD:nvcuvenc.dll /DELAYLOAD:nvcuvid.dll /DELAYLOAD:nvcuda.dll /DELAYLOAD:${CUDA_CUDART_DLL_NAME}")
endif(MSVC)


MITK_INSTALL(TARGETS niftknvapi)

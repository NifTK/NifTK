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

FIND_PACKAGE(CUDA REQUIRED)
 
SET(niftknvapi_SRCS
  niftkNVAPICommon.cpp
  glew/glew.c
  video/compress.cpp
  video/device.cpp
  video/deviceimpl.cpp
  video/frame.cpp
  video/sdiinput.cpp
  video/sdioutput.cpp
  video/rgba2nv12.cu
)

# with all the itk etc stuff, nvcc dies with "command line too long".
# so instead drop off all the not needed stuff and start clean.
# http://www.cmake.org/Bug/print_bug_page.php?bug_id=12198
SET_DIRECTORY_PROPERTIES( PROPERTIES INCLUDE_DIRECTORIES "" )

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/NVAPI/video)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/Code/Libs/NVAPI/glew)
INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_INCLUDE})

# ADD_LIBRARY(niftknvapi ${niftknvapi_SRCS})
CUDA_ADD_LIBRARY(niftknvapi ${niftknvapi_SRCS})

IF(BUILD_SHARED_LIBS)
  IF(WIN32)
    ADD_DEFINITIONS(-DNIFTKNVAPI_WINDOWS_EXPORT)
	ADD_DEFINITIONS(-DLIBVIDEO_BUILDING_DLL)
  ENDIF(WIN32)
ENDIF(BUILD_SHARED_LIBS)


GET_FILENAME_COMPONENT(cudaencoderpath ${CUDA_CUDART_LIBRARY} PATH)


TARGET_LINK_LIBRARIES(niftknvapi  
  ${NVAPI_LIBRARY}
  ${OPENGL_gl_LIBRARY}
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
  ${cudaencoderpath}/nvcuvenc.lib
  ${cudaencoderpath}/nvcuvid.lib
)

MITK_INSTALL(TARGETS niftknvapi)

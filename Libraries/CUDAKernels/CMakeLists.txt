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
 
set(niftkCUDAKernels_SRCS
  Example/niftkAddTwoIntegers.cu
  Example/niftkEdgeDetectionKernel.cu
  CameraCalibration/niftkUndistortionKernel.cu
  Image/niftkFlipImageKernel.cu
)

# with all the itk etc stuff, nvcc dies with "command line too long".
# so instead drop off all the not needed stuff and start clean.
# http://www.cmake.org/Bug/print_bug_page.php?bug_id=12198
set_directory_properties( PROPERTIES INCLUDE_DIRECTORIES "" )
include_directories(${CMAKE_SOURCE_DIR}/Libraries/CUDAKernels)
include_directories(${CMAKE_SOURCE_DIR}/Libraries/CUDAKernels/Common)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CUDA_TOOLKIT_INCLUDE})

if(APPLE)
  set(_tmp_cmake_cxx_flags ${CMAKE_CXX_FLAGS})
  string(REPLACE "-std=c++11" "" CMAKE_CXX_FLAGS ${_tmp_cmake_cxx_flags})
endif()

# FIXME: figure out what minimum sm-version we need!
# cmake is a real pain. it's impossible to get the quoting right for these options.
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--ptxas-options=-v;--generate-code arch=compute_30,code=sm_30;-lineinfo) #,code=\"sm_30,compute_30\"")

# we want symbols for the cuda code too. otherwise nsight doesn't produce much useful info.
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-G)
endif()

if(NOT WIN32 AND NOT APPLE)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
endif()

CUDA_ADD_LIBRARY(niftkCUDAKernels ${niftkCUDAKernels_SRCS})

if(BUILD_SHARED_LIBS)
  if(WIN32)
    # note: this one is also defined by cmake via nvcc!
    add_definitions(-DniftkCUDAKernels_EXPORTS)
  endif(WIN32)
endif(BUILD_SHARED_LIBS)


target_link_libraries(niftkCUDAKernels  
  ${CUDA_CUDA_LIBRARY}
  ${CUDA_CUDART_LIBRARY}
)

if(MSVC AND NIFTK_DELAYLOAD_CUDA AND CUDA_FOUND AND NIFTK_USE_CUDA)
  # all of the nvidia-driver/cuda dependencies should be delay-loaded
  #  so this library would at least load on a machine that doesnt have nv hardware
  set_target_properties(niftkCUDAKernels PROPERTIES LINK_FLAGS 
    "/DELAYLOAD:nvcuda.dll /DELAYLOAD:${CUDA_CUDART_DLL_NAME}")
endif()


MITK_INSTALL(TARGETS niftkCUDAKernels)

if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()

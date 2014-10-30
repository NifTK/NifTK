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

set(NIFTK_TEST_CUDA_LINK_LIBRARIES
  ${ITK_LIBRARIES}
  niftkCUDAKernels
)

#----------------------------------------------------------------------------------------------------------------------------
# Dont forget its:  add_test(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#----------------------------------------------------------------------------------------------------------------------------


set(CUDAKernelsTests_SRCS
  CameraCalibration/UndistortionKernelTests.cxx
)


include_directories(${CMAKE_SOURCE_DIR}/Code/Libs/CUDAKernels)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CUDA_TOOLKIT_INCLUDE})


add_executable(niftkCUDAKernelsTests niftkCUDAKernelsTests.cxx ${CUDAKernelsTests_SRCS})
target_link_libraries(niftkCUDAKernelsTests ${NIFTK_TEST_CUDA_LINK_LIBRARIES} )

set(CUDAKernels_TESTS ${CXX_TEST_PATH}/niftkCUDAKernelsTests)

add_test(CUDAKernels-UndistortionTest ${CUDAKernels_TESTS} niftkCUDAKernelsUndistortionTest )

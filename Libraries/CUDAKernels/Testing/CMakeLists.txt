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

set(CUDAKernelsTests_SRCS
)

set(NIFTK_TEST_CUDA_LINK_LIBRARIES
  ${ITK_LIBRARIES}
  ${Boost_LIBRARIES}
  niftkCUDAKernels
)

list(APPEND CUDAKernelsTests_SRCS Example/niftkAddTwoIntegersKernelTests.cxx)

# the test needs opencv for validation. but that is build automatically only for the igi config.
if(BUILD_NiftyIGI)
  set(CUDAKernelsTests_SRCS
    ${CUDAKernelsTests_SRCS}
    CameraCalibration/niftkUndistortionKernelTests.cxx
  )

  set(NIFTK_TEST_CUDA_LINK_LIBRARIES
    ${NIFTK_TEST_CUDA_LINK_LIBRARIES}
    ${OpenCV_LIBS}
  )
endif()

include_directories(${CMAKE_SOURCE_DIR}/Libraries/CUDAKernels)
include_directories(${CMAKE_SOURCE_DIR}/Libraries/CUDAKernels/Example)
include_directories(${CUDA_TOOLKIT_INCLUDE})
include_directories(${NIFTK_INCLUDE_DIRS_BUILD_TREE})

add_executable(niftkCUDAKernelsTests niftkCUDAKernelsTests.cxx ${CUDAKernelsTests_SRCS})
target_include_directories(niftkCUDAKernelsTests PRIVATE ${ITK_INCLUDE_DIRS})
target_link_libraries(niftkCUDAKernelsTests PRIVATE ${NIFTK_TEST_CUDA_LINK_LIBRARIES})

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Suppress warning because of a bug in the Boost gil library.
  set_target_properties(niftkCUDAKernelsTests PROPERTIES COMPILE_FLAGS "-Wno-c++11-narrowing")
endif()

set(CUDAKernels_TESTS ${CXX_TEST_PATH}/niftkCUDAKernelsTests)

set(TEMPORARY_OUTPUT ${NIFTK_BINARY_DIR}/Testing/Temporary)

#----------------------------------------------------------------------------------------------------------------------------
# Dont forget its:  add_test(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#----------------------------------------------------------------------------------------------------------------------------

add_test(CUDAKernels-UndistortionTest ${CUDAKernels_TESTS} niftkUndistortionKernelTests ${TEMPORARY_OUTPUT})
add_test(CUDAKernels-AddTwoIntegers ${CUDAKernels_TESTS} niftkAddTwoIntegersKernelTests )

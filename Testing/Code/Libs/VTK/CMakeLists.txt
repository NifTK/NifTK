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

if(NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  # OpenMP is disabled when the compiler is based on Clang
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif(OPENMP_FOUND)
endif(NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")

ADD_DEFINITIONS(-DUSE_NIFTYREG)
LINK_DIRECTORIES(${NIFTYREG_LIBRARY_DIR})
INCLUDE_DIRECTORIES(${NIFTYREG_INCLUDE_DIR})

SET(NIFTK_TEST_EXT_VTK_LINK_LIBRARIES 
  ITKIO
  vtkIO
  vtkImaging
  niftkVTK
  ${Boost_LIBRARIES}
  ${NIFTYREG_LIBRARIES}
  )

#----------------------------------------------------------------------------------------------------------------------------
# Dont forget its:  ADD_TEST(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#----------------------------------------------------------------------------------------------------------------------------
SET(TEMPORARY_OUTPUT ${NIFTK_BINARY_DIR}/Testing/Temporary)

#################################################################################
# Build instructions for Unit Tests.
#################################################################################
SET( VTKUnitTests_SRCS
  VTKLoadSaveMatrix4x4Test.cxx
)

ADD_EXECUTABLE(VTKUnitTests VTKUnitTests.cxx ${VTKUnitTests_SRCS})
TARGET_LINK_LIBRARIES(VTKUnitTests ${NIFTK_TEST_EXT_VTK_LINK_LIBRARIES} )

SET(VTK_UNIT_TESTS ${CXX_TEST_PATH}/VTKUnitTests)

ADD_TEST(VTK-LoadSaveMatrix4x4Test ${VTK_UNIT_TESTS} VTKLoadSaveMatrix4x4Test ${TEMPORARY_OUTPUT}/VTKLoadSaveMatrix4x4Test.txt )

#################################################################################
# Build instructions for Integration Tests.
#################################################################################
SET( VTKIntegrationTests_SRCS
  VTKIterativeClosestPointTest.cxx
)

ADD_EXECUTABLE(VTKIntegrationTests VTKIntegrationTests.cxx ${VTKIntegrationTests_SRCS})
TARGET_LINK_LIBRARIES(VTKIntegrationTests ${NIFTK_TEST_EXT_VTK_LINK_LIBRARIES} )

SET(VTK_INTEGRATION_TESTS ${CXX_TEST_PATH}/VTKIntegrationTests)

ADD_TEST(SLS-VTK-ICP-Test ${VTK_INTEGRATION_TESTS} VTKIterativeClosestPointTest ${INPUT_DATA}/liver.vtk ${INPUT_DATA}/liver.vtk noperturb)
IF(BUILD_SLS_TESTING)
  ADD_TEST(SLS-VTK-ICP-Perturb-Test ${VTK_INTEGRATION_TESTS} VTKIterativeClosestPointTest ${INPUT_DATA}/liver.vtk ${INPUT_DATA}/liver.vtk perturb)
  ADD_TEST(SLS-VTK-ICP-Rep-Test ${VTK_INTEGRATION_TESTS} VTKIterativeClosestPointRepeatTest ${INPUT_DATA}/maskedliver.vtk ${INPUT_DATA}/liver.vtk)
ENDIF()



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

SET(NIFTK_TEST_EXT_VTK_LINK_LIBRARIES 
  ITKIO
  vtkIO
  vtkImaging
  niftkVTK
  ${Boost_LIBRARIES}
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
     VTKTextFileReaderTests.cxx
     VTKDistanceToSurfaceTests.cxx
)

ADD_EXECUTABLE(VTKIntegrationTests VTKIntegrationTests.cxx ${VTKIntegrationTests_SRCS})
TARGET_LINK_LIBRARIES(VTKIntegrationTests ${NIFTK_TEST_EXT_VTK_LINK_LIBRARIES} )

SET(VTK_INTEGRATION_TESTS ${CXX_TEST_PATH}/VTKIntegrationTests)

ADD_TEST(VTK-4Point-Reader-Test ${VTK_INTEGRATION_TESTS} VTK4PointReaderTest ${INPUT_DATA}/4points.txt)
ADD_TEST(VTK-3Point-Reader-Test ${VTK_INTEGRATION_TESTS} VTK3PointReaderTest ${INPUT_DATA}/3points.txt)
ADD_TEST(VTK-Distance-To-Surface-PolyData-Test ${VTK_INTEGRATION_TESTS} VTKDistanceToSurfaceTest ${INPUT_DATA}/3points.txt  ${INPUT_DATA}/liver.vtk ${INPUT_DATA}/PointsToSurfaceDist.txt)
ADD_TEST(VTK-Distance-To-Surface-Single-Test ${VTK_INTEGRATION_TESTS} VTKDistanceToSurfaceTestSinglePoint 5.4 2.20 1.134 ${INPUT_DATA}/liver.vtk 146.72)
ADD_TEST(SLS-VTK-ICP-Test ${VTK_INTEGRATION_TESTS} VTKIterativeClosestPointTest ${INPUT_DATA}/liver.vtk ${INPUT_DATA}/liver.vtk noperturb)

IF(BUILD_SLS_TESTING)
  ADD_TEST(SLS-VTK-ICP-Perturb-Test ${VTK_INTEGRATION_TESTS} VTKIterativeClosestPointTest ${INPUT_DATA}/liver.vtk ${INPUT_DATA}/liver.vtk perturb)
  ADD_TEST(SLS-VTK-ICP-Rep-Test ${VTK_INTEGRATION_TESTS} VTKIterativeClosestPointRepeatTest ${INPUT_DATA}/maskedliver.vtk ${INPUT_DATA}/liver.vtk)
ENDIF()



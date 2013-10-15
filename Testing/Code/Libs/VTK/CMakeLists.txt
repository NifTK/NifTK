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

set(NIFTK_TEST_EXT_VTK_LINK_LIBRARIES
  ${ITK_LIBRARIES}
  vtkIO
  vtkImaging
  niftkVTK
  ${Boost_LIBRARIES}
  )

#----------------------------------------------------------------------------------------------------------------------------
# Dont forget its:  add_test(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#----------------------------------------------------------------------------------------------------------------------------
set(TEMPORARY_OUTPUT ${NIFTK_BINARY_DIR}/Testing/Temporary)

#################################################################################
# Build instructions for Unit Tests.
#################################################################################
set(VTKUnitTests_SRCS
  niftkVTKLoadSaveMatrix4x4Test.cxx
)

add_executable(niftkVTKUnitTests niftkVTKUnitTests.cxx ${VTKUnitTests_SRCS})
target_link_libraries(niftkVTKUnitTests ${NIFTK_TEST_EXT_VTK_LINK_LIBRARIES} )

set(VTK_UNIT_TESTS ${CXX_TEST_PATH}/niftkVTKUnitTests)

add_test(VTK-LoadSaveMatrix4x4Test ${VTK_UNIT_TESTS} niftkVTKLoadSaveMatrix4x4Test ${TEMPORARY_OUTPUT}/VTKLoadSaveMatrix4x4Test.txt )

#################################################################################
# Build instructions for Integration Tests.
#################################################################################
set(VTKIntegrationTests_SRCS
  niftkVTKIterativeClosestPointTest.cxx
  niftkVTKTextFileReaderTests.cxx
  niftkVTKDistanceToSurfaceTests.cxx
)

add_executable(niftkVTKIntegrationTests niftkVTKIntegrationTests.cxx ${VTKIntegrationTests_SRCS})
target_link_libraries(niftkVTKIntegrationTests ${NIFTK_TEST_EXT_VTK_LINK_LIBRARIES} )

set(VTK_INTEGRATION_TESTS ${CXX_TEST_PATH}/niftkVTKIntegrationTests)

add_test(VTK-4Point-Reader-Test ${VTK_INTEGRATION_TESTS} niftkVTK4PointReaderTest ${INPUT_DATA}/4points.txt)
add_test(VTK-3Point-Reader-Test ${VTK_INTEGRATION_TESTS} niftkVTK3PointReaderTest ${INPUT_DATA}/3points.txt)
add_test(VTK-Distance-To-Surface-PolyData-Test ${VTK_INTEGRATION_TESTS} niftkVTKDistanceToSurfaceTest ${INPUT_DATA}/3points.txt  ${INPUT_DATA}/liver.vtk ${INPUT_DATA}/PointsToSurfaceDist.txt)
add_test(VTK-Distance-To-Surface-Single-Test ${VTK_INTEGRATION_TESTS} niftkVTKDistanceToSurfaceTestSinglePoint 5.4 2.20 1.134 ${INPUT_DATA}/liver.vtk 146.72)
add_test(SLS-VTK-ICP-Test ${VTK_INTEGRATION_TESTS} niftkVTKIterativeClosestPointTest ${INPUT_DATA}/liver.vtk ${INPUT_DATA}/liver.vtk noperturb)
if(BUILD_SLS_TESTING)
  add_test(SLS-VTK-ICP-Perturb-Test ${VTK_INTEGRATION_TESTS} niftkVTKIterativeClosestPointTest ${INPUT_DATA}/liver.vtk ${INPUT_DATA}/liver.vtk perturb)
  add_test(SLS-VTK-ICP-Rep-Test ${VTK_INTEGRATION_TESTS} niftkVTKIterativeClosestPointRepeatTest ${INPUT_DATA}/maskedliver.vtk ${INPUT_DATA}/liver.vtk)
endif()

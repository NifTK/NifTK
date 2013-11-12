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

set(NIFTK_TEST_EXT_ITK_COMMON_LINK_LIBRARIES
  niftkcommon
  niftkITK
  ${ITK_LIBRARIES}
  ${Boost_LIBRARIES}
  )

# This is the name of the actual executable that gets run.
set(ITK_COMMON_UNIT_TESTS ${CXX_TEST_PATH}/ITKCommonUnitTests)
set(TEMPORARY_OUTPUT ${NIFTK_BINARY_DIR}/Testing/Temporary)

#----------------------------------
# Dont forget its:  add_test(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#--------------------------------------------------------------------------------------

#################################################################################
# First we test the transformations, as this is purely coordinate/geometry based.
#################################################################################

add_test(Common-CheckDims-2D ${ITK_COMMON_UNIT_TESTS} CheckImageDimensionalityTest ${INPUT_DATA}/cte_20_x_20.png 2 )
add_test(Common-CheckDims-3D ${ITK_COMMON_UNIT_TESTS} CheckImageDimensionalityTest ${INPUT_DATA}/ADNI-AD-098_S_0149-Scr-03300.img 3 )
add_test(Common-ReceptorMember ${ITK_COMMON_UNIT_TESTS} ReceptorMemberCommandTest )

#################################################################################
# Build instructions.
#################################################################################
set(ITKCommonUnitTests_SRCS
  CheckImageDimensionalityTest.cxx
  ReceptorMemberCommandTest.cxx
)

add_executable(ITKCommonUnitTests ITKCommonUnitTests.cxx ${ITKCommonUnitTests_SRCS})
target_link_libraries(ITKCommonUnitTests ${NIFTK_TEST_EXT_ITK_COMMON_LINK_LIBRARIES} )

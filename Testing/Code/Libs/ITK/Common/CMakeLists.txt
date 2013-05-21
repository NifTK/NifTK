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

SET(NIFTK_TEST_EXT_ITK_COMMON_LINK_LIBRARIES 
  niftkcommon 
  niftkITK 
  ITKCommon 
  ITKIO  
  ${Boost_LIBRARIES}
  )

# This is the name of the actual executable that gets run.  
SET(ITK_COMMON_UNIT_TESTS ${CXX_TEST_PATH}/ITKCommonUnitTests)
SET(TEMPORARY_OUTPUT ${NIFTK_BINARY_DIR}/Testing/Temporary)

#----------------------------------
# Dont forget its:  ADD_TEST(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#--------------------------------------------------------------------------------------

#################################################################################
# First we test the transformations, as this is purely coordinate/geometry based.
#################################################################################

ADD_TEST(Common-CheckDims-2D ${ITK_COMMON_UNIT_TESTS} CheckImageDimensionalityTest ${INPUT_DATA}/cte_20_x_20.png 2 )
ADD_TEST(Common-CheckDims-3D ${ITK_COMMON_UNIT_TESTS} CheckImageDimensionalityTest ${INPUT_DATA}/ADNI-AD-098_S_0149-Scr-03300.img 3 )
ADD_TEST(Common-ReceptorMember ${ITK_COMMON_UNIT_TESTS} ReceptorMemberCommandTest )

#################################################################################
# Build instructions.
#################################################################################
SET( ITKCommonUnitTests_SRCS
CheckImageDimensionalityTest.cxx
ReceptorMemberCommandTest.cxx
)

ADD_EXECUTABLE(ITKCommonUnitTests ITKCommonUnitTests.cxx ${ITKCommonUnitTests_SRCS})
TARGET_LINK_LIBRARIES(ITKCommonUnitTests ${NIFTK_TEST_EXT_ITK_COMMON_LINK_LIBRARIES} )

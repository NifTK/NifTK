#/*================================================================================
#
#  NifTK: An image processing toolkit jointly developed by the
#              Dementia Research Centre, and the Centre For Medical Image Computing
#              at University College London.
#  
#  See:        http://dementia.ion.ucl.ac.uk/
#              http://cmic.cs.ucl.ac.uk/
#              http://www.ucl.ac.uk/
#
#  Copyright (c) UCL : See LICENSE.txt in the top level directory for details. 
#
#  Last Changed      : $LastChangedDate: 2011-09-28 16:00:43 +0100 (Wed, 28 Sep 2011) $ 
#  Revision          : $Revision: 7385 $
#  Last modified by  : $Author: mjc $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

SET(NIFTK_TEST_EXT_ITK_COMMON_LINK_LIBRARIES 
  niftkcommon 
  niftkITK 
  ITKCommon 
  ITKIO  
  ${Boost_LIBRARIES}
  )

# This is the name of the actual executable that gets run.  
SET(COMMON_INTEGRATION_TESTS ${CXX_TEST_PATH}/CommonIntegrationTests)
SET(TEMPORARY_OUTPUT ${NIFTK_BINARY_DIR}/Testing/Temporary)
SET(BASELINE ${NIFTK_DATA_ROOT}/Baseline)

#----------------------------------
# Dont forget its:  ADD_TEST(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#--------------------------------------------------------------------------------------

#################################################################################
# First we test the transformations, as this is purely coordinate/geometry based.
#################################################################################

ADD_TEST(Common-CheckDims-2D ${COMMON_INTEGRATION_TESTS} CheckImageDimensionalityTest ${INPUT_DATA}/cte_20_x_20.png 2 )
ADD_TEST(Common-CheckDims-3D ${COMMON_INTEGRATION_TESTS} CheckImageDimensionalityTest ${INPUT_DATA}/ADNI-AD-098_S_0149-Scr-03300.img 3 )
ADD_TEST(Common-ReceptorMember ${COMMON_INTEGRATION_TESTS} ReceptorMemberCommandTest )

#################################################################################
# Build instructions.
#################################################################################
SET( CommonIntegrationTests_SRCS
CheckImageDimensionalityTest.cxx
ReceptorMemberCommandTest.cxx
)

ADD_EXECUTABLE(CommonIntegrationTests CommonIntegrationTests.cxx ${CommonIntegrationTests_SRCS})
TARGET_LINK_LIBRARIES(CommonIntegrationTests ${NIFTK_TEST_EXT_ITK_COMMON_LINK_LIBRARIES} )

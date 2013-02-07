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

SET(NIFTK_TEST_EXT_MIDAS_LINK_LIBRARIES 
  niftkmidas
  niftkcommon
  ITKCommon
  ITKIO
  ${Boost_LIBRARIES}
  )

# This is the name of the actual executable that gets run.  
SET(MIDAS_INTEGRATION_TESTS ${CXX_TEST_PATH}/MidasIntegrationTests)
SET(TEMPORARY_OUTPUT ${NIFTK_BINARY_DIR}/Testing/Temporary)

#----------------------------------
# Dont forget its:  ADD_TEST(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#--------------------------------------------------------------------------------------

ADD_TEST(MIDAS-LoadRegionTest ${MIDAS_INTEGRATION_TESTS} LoadRegionTest ${INPUT_DATA}/ADNI-AD-098_S_0149-Scr-01728_REG.img ${INPUT_DATA}/ADNI-AD-098_S_0149-Scr-01728_REG 10)

#################################################################################
# Build instructions.
#################################################################################
SET( MidasIntegrationTests_SRCS
LoadRegionTest.cxx
)

ADD_EXECUTABLE(MidasIntegrationTests MidasIntegrationTests.cxx ${MidasIntegrationTests_SRCS})
TARGET_LINK_LIBRARIES(MidasIntegrationTests ${NIFTK_TEST_EXT_MIDAS_LINK_LIBRARIES} )

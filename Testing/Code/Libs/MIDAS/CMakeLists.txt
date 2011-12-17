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
#  Last Changed      : $LastChangedDate: 2010-08-11 08:28:23 +0100 (Wed, 11 Aug 2010) $ 
#  Revision          : $Revision: 7310 $
#  Last modified by  : $Author: ad $
#
#  Original author   : m.clarkson@cs.ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

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
SET(BASELINE ${NIFTK_DATA_ROOT}/Baseline)

#----------------------------------
# Dont forget its:  ADD_TEST(<test name (unique to this file) > <exe name> <test name from C++ file> <argument1> <argument2>
#--------------------------------------------------------------------------------------

ADD_TEST(LoadRegionTest ${MIDAS_INTEGRATION_TESTS} LoadRegionTest ${INPUT_DATA}/ADNI-AD-098_S_0149-Scr-01728_REG.img ${INPUT_DATA}/ADNI-AD-098_S_0149-Scr-01728_REG 10)

#################################################################################
# Build instructions.
#################################################################################
SET( MidasIntegrationTests_SRCS
LoadRegionTest.cxx
)

ADD_EXECUTABLE(MidasIntegrationTests MidasIntegrationTests.cxx ${MidasIntegrationTests_SRCS})
TARGET_LINK_LIBRARIES(MidasIntegrationTests ${NIFTK_TEST_EXT_MIDAS_LINK_LIBRARIES} )

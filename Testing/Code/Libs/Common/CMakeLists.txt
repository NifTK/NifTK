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
#  Last Changed      : $LastChangedDate: 2011-09-14 11:37:54 +0100 (Wed, 14 Sep 2011) $ 
#  Revision          : $Revision: 7310 $
#  Last modified by  : $Author: ad $
#
#  Original author   : m.clarkson@ucl.ac.uk
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notices for more information.
#
#=================================================================================*/

SET(NIFTK_COMMON_TEST_LINK_LIBRARIES 
  niftkcommon 
  ITKCommon 
  ITKIO 
  ${Boost_LIBRARIES}
  )
  
#--------------------------------------------------------------------------------------
# Dont forget its:  ADD_TEST(<test name> <exe name> <test name> <argument1> <argument2>
#--------------------------------------------------------------------------------------
ADD_TEST(Conv-Utils-01 ${EXECUTABLE_OUTPUT_PATH}/CommonUnitTests ConversionUtilsTest 1)
ADD_TEST(Conv-Utils-02 ${EXECUTABLE_OUTPUT_PATH}/CommonUnitTests ConversionUtilsTest 2)
ADD_TEST(Conv-Utils-03 ${EXECUTABLE_OUTPUT_PATH}/CommonUnitTests ConversionUtilsTest 3)
ADD_TEST(Conv-Utils-04 ${EXECUTABLE_OUTPUT_PATH}/CommonUnitTests ConversionUtilsTest 4)
ADD_TEST(Conv-Utils-05 ${EXECUTABLE_OUTPUT_PATH}/CommonUnitTests ConversionUtilsTest 5)
ADD_TEST(Conv-Utils-06 ${EXECUTABLE_OUTPUT_PATH}/CommonUnitTests ConversionUtilsTest 6)
ADD_TEST(Deliberate-Memory-Leak ${EXECUTABLE_OUTPUT_PATH}/CommonUnitTests DeliberateMemoryLeakTest )

SET( CommonUnitTests_SRCS
ConversionUtilsTest.cxx
DeliberateMemoryLeakTest.cxx
)

ADD_EXECUTABLE(CommonUnitTests CommonUnitTests.cxx ${CommonUnitTests_SRCS})
TARGET_LINK_LIBRARIES(CommonUnitTests ${NIFTK_COMMON_TEST_LINK_LIBRARIES} )

SET(NIFTK_FILE_TEST_LINK_LIBRARIES 
  niftkcommon 
  ITKCommon 
  ITKIO
  ${Boost_LIBRARIES}
  )

  
#--------------------------------------------------------------------------------------
# Dont forget its:  ADD_TEST(<test name> <exe name> <test name> <argument1> <argument2>
#--------------------------------------------------------------------------------------
ADD_TEST(File-Helper-1 ${EXECUTABLE_OUTPUT_PATH}/FileUnitTests FileHelperTest 1)
ADD_TEST(File-Helper-2 ${EXECUTABLE_OUTPUT_PATH}/FileUnitTests FileHelperTest 2)
ADD_TEST(File-Helper-3 ${EXECUTABLE_OUTPUT_PATH}/FileUnitTests FileHelperTest 3)
ADD_TEST(File-Helper-4 ${EXECUTABLE_OUTPUT_PATH}/FileUnitTests FileHelperTest 4)
ADD_TEST(File-Helper-5 ${EXECUTABLE_OUTPUT_PATH}/FileUnitTests FileHelperTest 5)
ADD_TEST(File-Helper-6 ${EXECUTABLE_OUTPUT_PATH}/FileUnitTests FileHelperTest 6)
ADD_TEST(File-Helper-7 ${EXECUTABLE_OUTPUT_PATH}/FileUnitTests FileHelperTest 7)

SET( FileUnitTests_SRCS
FileHelperTest.cxx
)

ADD_EXECUTABLE(FileUnitTests FileUnitTests.cxx ${FileUnitTests_SRCS})
TARGET_LINK_LIBRARIES(FileUnitTests ${NIFTK_FILE_TEST_LINK_LIBRARIES} )

#--------------------------------------------------------------------------------------
# Unit tests for meshing module (only compiled if meshing is turned on)
#--------------------------------------------------------------------------------------
IF (BUILD_MESHING)
  ADD_SUBDIRECTORY(Meshing)
ENDIF (BUILD_MESHING)
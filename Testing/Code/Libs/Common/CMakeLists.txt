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

set(NIFTK_COMMON_TEST_LINK_LIBRARIES 
  niftkcommon 
  ITKCommon 
  ITKIO 
  ${Boost_LIBRARIES}
  )
  
#--------------------------------------------------------------------------------------
# Dont forget its:  add_test(<test name> <exe name> <test name> <argument1> <argument2>
#--------------------------------------------------------------------------------------
add_test(Conv-Utils-01 ${EXECUTABLE_OUTPUT_PATH}/CommonUnitTests ConversionUtilsTest 1)
add_test(Conv-Utils-02 ${EXECUTABLE_OUTPUT_PATH}/CommonUnitTests ConversionUtilsTest 2)
add_test(Conv-Utils-03 ${EXECUTABLE_OUTPUT_PATH}/CommonUnitTests ConversionUtilsTest 3)
add_test(Conv-Utils-04 ${EXECUTABLE_OUTPUT_PATH}/CommonUnitTests ConversionUtilsTest 4)
add_test(Conv-Utils-05 ${EXECUTABLE_OUTPUT_PATH}/CommonUnitTests ConversionUtilsTest 5)
add_test(Conv-Utils-06 ${EXECUTABLE_OUTPUT_PATH}/CommonUnitTests ConversionUtilsTest 6)
add_test(Deliberate-Memory-Leak ${EXECUTABLE_OUTPUT_PATH}/CommonUnitTests DeliberateMemoryLeakTest )

set( CommonUnitTests_SRCS
ConversionUtilsTest.cxx
DeliberateMemoryLeakTest.cxx
)

add_executable(CommonUnitTests CommonUnitTests.cxx ${CommonUnitTests_SRCS})
target_link_libraries(CommonUnitTests ${NIFTK_COMMON_TEST_LINK_LIBRARIES} )

set(NIFTK_FILE_TEST_LINK_LIBRARIES 
  niftkcommon 
  ITKCommon 
  ITKIO
  ${Boost_LIBRARIES}
  )

  
#--------------------------------------------------------------------------------------
# Dont forget its:  add_test(<test name> <exe name> <test name> <argument1> <argument2>
#--------------------------------------------------------------------------------------
add_test(File-Helper-1 ${EXECUTABLE_OUTPUT_PATH}/FileUnitTests FileHelperTest 1)
add_test(File-Helper-2 ${EXECUTABLE_OUTPUT_PATH}/FileUnitTests FileHelperTest 2)
add_test(File-Helper-3 ${EXECUTABLE_OUTPUT_PATH}/FileUnitTests FileHelperTest 3)
add_test(File-Helper-4 ${EXECUTABLE_OUTPUT_PATH}/FileUnitTests FileHelperTest 4)
add_test(File-Helper-5 ${EXECUTABLE_OUTPUT_PATH}/FileUnitTests FileHelperTest 5)
add_test(File-Helper-6 ${EXECUTABLE_OUTPUT_PATH}/FileUnitTests FileHelperTest 6)
add_test(File-Helper-7 ${EXECUTABLE_OUTPUT_PATH}/FileUnitTests FileHelperTest 7)

set( FileUnitTests_SRCS
FileHelperTest.cxx
)

add_executable(FileUnitTests FileUnitTests.cxx ${FileUnitTests_SRCS})
target_link_libraries(FileUnitTests ${NIFTK_FILE_TEST_LINK_LIBRARIES} )

#--------------------------------------------------------------------------------------
# Unit tests for meshing module (only compiled if meshing is turned on)
#--------------------------------------------------------------------------------------
if (BUILD_MESHING)
  add_subdirectory(Meshing)
endif (BUILD_MESHING)
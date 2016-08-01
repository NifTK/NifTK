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

set(NIFTK_COMMON_TEST_INCLUDE_DIRS
  ${ITK_INCLUDE_DIRS}
  )

set(NIFTK_COMMON_TEST_LINK_LIBRARIES
  niftkcommon
  ${ITK_LIBRARIES}
  ${Boost_LIBRARIES}
  )

#--------------------------------------------------------------------------------------
# Dont forget its:  add_test(<test name> <exe name> <test name> <argument1> <argument2>
#--------------------------------------------------------------------------------------
add_test(Conv-Utils-01 ${EXECUTABLE_OUTPUT_PATH}/niftkCommonUnitTests niftkConversionUtilsTest 1)
add_test(Conv-Utils-02 ${EXECUTABLE_OUTPUT_PATH}/niftkCommonUnitTests niftkConversionUtilsTest 2)
add_test(Conv-Utils-03 ${EXECUTABLE_OUTPUT_PATH}/niftkCommonUnitTests niftkConversionUtilsTest 3)
add_test(Conv-Utils-04 ${EXECUTABLE_OUTPUT_PATH}/niftkCommonUnitTests niftkConversionUtilsTest 4)
add_test(Conv-Utils-05 ${EXECUTABLE_OUTPUT_PATH}/niftkCommonUnitTests niftkConversionUtilsTest 5)
add_test(Conv-Utils-06 ${EXECUTABLE_OUTPUT_PATH}/niftkCommonUnitTests niftkConversionUtilsTest 6)
add_test(Deliberate-Memory-Leak ${EXECUTABLE_OUTPUT_PATH}/niftkCommonUnitTests niftkDeliberateMemoryLeakTest )

set(CommonUnitTests_SRCS
  niftkConversionUtilsTest.cxx
  niftkDeliberateMemoryLeakTest.cxx
)

add_executable(niftkCommonUnitTests niftkCommonUnitTests.cxx ${CommonUnitTests_SRCS})
target_include_directories(niftkCommonUnitTests PRIVATE ${NIFTK_COMMON_TEST_INCLUDE_DIRS})
target_link_libraries(niftkCommonUnitTests PRIVATE ${NIFTK_COMMON_TEST_LINK_LIBRARIES} )

set(NIFTK_FILE_TEST_INCLUDE_DIRS
  ${ITK_INCLUDE_DIRS}
  )

set(NIFTK_FILE_TEST_LINK_LIBRARIES
  niftkcommon
  ${ITK_LIBRARIES}
  ${Boost_LIBRARIES}
  )


#--------------------------------------------------------------------------------------
# Dont forget its:  add_test(<test name> <exe name> <test name> <argument1> <argument2>
#--------------------------------------------------------------------------------------
add_test(File-Helper-1 ${EXECUTABLE_OUTPUT_PATH}/niftkFileUnitTests niftkFileHelperTest 1)
add_test(File-Helper-2 ${EXECUTABLE_OUTPUT_PATH}/niftkFileUnitTests niftkFileHelperTest 2)
add_test(File-Helper-3 ${EXECUTABLE_OUTPUT_PATH}/niftkFileUnitTests niftkFileHelperTest 3)
add_test(File-Helper-4 ${EXECUTABLE_OUTPUT_PATH}/niftkFileUnitTests niftkFileHelperTest 4)
add_test(File-Helper-5 ${EXECUTABLE_OUTPUT_PATH}/niftkFileUnitTests niftkFileHelperTest 5)
add_test(File-Helper-6 ${EXECUTABLE_OUTPUT_PATH}/niftkFileUnitTests niftkFileHelperTest 6)
add_test(File-Helper-7 ${EXECUTABLE_OUTPUT_PATH}/niftkFileUnitTests niftkFileHelperTest 7)
add_test(File-Helper-8 ${EXECUTABLE_OUTPUT_PATH}/niftkFileUnitTests niftkFileHelperTest 8)
add_test(File-Helper-9 ${EXECUTABLE_OUTPUT_PATH}/niftkFileUnitTests niftkFileHelperTest 9)
add_test(File-Helper-10 ${EXECUTABLE_OUTPUT_PATH}/niftkFileUnitTests niftkFileHelperTest 10)
add_test(File-Helper-11 ${EXECUTABLE_OUTPUT_PATH}/niftkFileUnitTests niftkFileHelperTest 11)
add_test(File-Helper-12 ${EXECUTABLE_OUTPUT_PATH}/niftkFileUnitTests niftkFileHelperTest 12 ${INPUT_DATA}/LapTrackingAccuracy/2013-07-26-17-00-36-631)
add_test(File-Helper-13 ${EXECUTABLE_OUTPUT_PATH}/niftkFileUnitTests niftkFileHelperTest 13 ${INPUT_DATA}/LapTrackingAccuracy/2013-07-26-17-00-36-631)
add_test(FixedLengthFileReader ${EXECUTABLE_OUTPUT_PATH}/niftkFileUnitTests niftkFixedLengthFileReaderTest ${INPUT_DATA}/AprilTagUnitTest/idmat.4x4)

set(FileUnitTests_SRCS
  niftkFileHelperTest.cxx
  niftkFixedLengthFileReaderTest.cxx
)

add_executable(niftkFileUnitTests niftkFileUnitTests.cxx ${FileUnitTests_SRCS})
target_include_directories(niftkFileUnitTests PRIVATE ${NIFTK_FILE_TEST_INCLUDE_DIRS} )
target_link_libraries(niftkFileUnitTests PRIVATE ${NIFTK_FILE_TEST_LINK_LIBRARIES} )

#--------------------------------------------------------------------------------------
# Unit tests for meshing module (only compiled if meshing is turned on)
#--------------------------------------------------------------------------------------
if (BUILD_MESHING)
  add_subdirectory(Meshing)
endif (BUILD_MESHING)
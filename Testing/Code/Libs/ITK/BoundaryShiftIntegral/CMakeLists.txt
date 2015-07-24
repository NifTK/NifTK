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

project(itkBSITesting)

# Input data.
set(INPUTDATA ${NIFTK_DATA_DIR}/BSIInput)
# Ouput data.
set(TEMPORARY_OUTPUT ${NIFTK_BINARY_DIR}/Testing/Temporary)

# Set up the tests with the correct executables.
set(LOCAL_TESTS ${CXX_TEST_PATH}/itkBSILocalTests)

add_test(BSI-itkMultipleDilateImageFilterTest ${LOCAL_TESTS} itkMultipleDilateErodeImageFilterTest)
add_test(BSI-itkBinaryIntersectWithPaddingImageFilterTest ${LOCAL_TESTS} itkBinaryIntersectWithPaddingImageFilterTest)
add_test(BSI-itkBinaryUnionWithPaddingImageFilterTest ${LOCAL_TESTS} itkBinaryUnionWithPaddingImageFilterTest)
add_test(BSI-itkIntensityNormalisationCalculatorTest ${LOCAL_TESTS} itkIntensityNormalisationCalculatorTest
         ${INPUTDATA}/134055125-623-1.hdr
         ${INPUTDATA}/29790120-1200-1.hdr
         ${INPUTDATA}/134055125-623-1_test_0.hdr
         ${INPUTDATA}/134055125-623-1_test_1.hdr
         3.587049166392765e+02
         6.823141922906915e+02)

add_test(BSI-itkSimpleKMeansClusteringImageFilterTest ${LOCAL_TESTS} itkSimpleKMeansClusteringImageFilterTest
         ${INPUTDATA}/56970125-005-1.hdr
         ${INPUTDATA}/Lai_56970125_1188319593_mask.hdr
         ${INPUTDATA}/85680125-005-1.hdr
         ${INPUTDATA}/Tra_85680125_1188322273_mask.hdr
         ${TEMPORARY_OUTPUT}/knbsi-kmeans.hdr
         280.391 678.648 953.249
         113.429 89.674 75.0122)

if(NOT WIN32 AND BUILD_COMMAND_LINE_PROGRAMS)
  add_test(BSI-itkKN-BSITest1  ${itkBSITesting_SOURCE_DIR}/itkKN-BSITests.sh
         ${CXX_TEST_PATH}/niftkKMeansWindowWithLinearRegressionNormalisationBSI
         ${INPUTDATA}/56970125-005-1.hdr
         ${INPUTDATA}/Lai_56970125_1188319593_mask.hdr
         ${INPUTDATA}/85680125-005-1.hdr
         ${INPUTDATA}/Tra_85680125_1188322273_mask.hdr
         ${INPUTDATA}/56970125-005-1.hdr
         ${INPUTDATA}/Lai_56970125_1188319593_mask.hdr
         ${INPUTDATA}/85680125-005-1.hdr
         ${INPUTDATA}/Tra_85680125_1188322273_mask.hdr
         1 1 3 -1 -1
         ${TEMPORARY_OUTPUT}/baseline-seg.hdr
         ${TEMPORARY_OUTPUT}/repeat-seg.hdr
         ${TEMPORARY_OUTPUT}/repeat-normalised.hdr
         ${TEMPORARY_OUTPUT}/kn-bsi-test-output.txt)

  add_test(BSI-itkKN-BSITest2 ${CMAKE_COMMAND} -E compare_files ${INPUTDATA}/kn-bsi-test-output.txt ${TEMPORARY_OUTPUT}/kn-bsi-test-output.txt)
  set_property(TEST BSI-itkKN-BSITest2 APPEND PROPERTY DEPENDS BSI-itkKN-BSITest1 )

  add_test(BSI-itkKN-BSITest3 ${CMAKE_COMMAND} -E compare_files ${INPUTDATA}/repeat-normalised.img ${TEMPORARY_OUTPUT}/repeat-normalised.img)
  set_property(TEST BSI-itkKN-BSITest3 APPEND PROPERTY DEPENDS BSI-itkKN-BSITest1 )

  #add_test(itkKN-BSITest4 ${CMAKE_COMMAND} -E compare_files ${INPUTDATA}/repeat-normalised.hdr ${TEMPORARY_OUTPUT}/repeat-normalised.hdr)
endif()

add_test(BSI-itkBoundaryShiftIntegralTest0 ${LOCAL_TESTS} itkBoundaryShiftIntegralTest
         ${INPUTDATA}/56970125-005-1.hdr
         ${INPUTDATA}/Lai_56970125_1188319593_mask.hdr
         ${INPUTDATA}/85680125-005-1.hdr
         ${INPUTDATA}/Tra_85680125_1188322273_mask.hdr
         ${INPUTDATA}/56970125-005-1.hdr
         ${INPUTDATA}/Lai_56970125_1188319593_mask.hdr
         ${INPUTDATA}/85680125-005-1.hdr
         ${INPUTDATA}/Tra_85680125_1188322273_mask.hdr
         1
         1
         0.25
         0.75
         ""
         5.62215631041469)

# Build the executables.
add_executable(itkBSILocalTests itkLocalTests.cxx
               itkBinaryIntersectWithPaddingImageFilterTest.cxx
               itkBinaryUnionWithPaddingImageFilterTest.cxx
               itkIntensityNormalisationCalculatorTest.cxx
               itkBoundaryShiftIntegralTest.cxx
               itkMultipleDilateImageFilterTest.cxx
               itkSimpleKMeansClusteringImageFilterTest.cxx)

target_include_directories(itkBSILocalTests PRIVATE ${ITK_INCLUDE_DIRS})
target_link_libraries(itkBSILocalTests PRIVATE ${ITK_LIBRARIES})

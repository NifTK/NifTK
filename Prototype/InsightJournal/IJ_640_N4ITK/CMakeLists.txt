cmake_minimum_required(VERSION 2.4)

PROJECT( N3MRIBiasFieldCorrectionImageFilter )

#The following lines are required to use Dart
ENABLE_TESTING()
INCLUDE(Dart)

# Set up ITK
IF(USE_ITK)
  FIND_PACKAGE(ITK)
  IF(ITK_FOUND)
    INCLUDE(${ITK_USE_FILE})
  ELSE(ITK_FOUND)
    MESSAGE(FATAL_ERROR
      "Cannot build without ITK.  Please set ITK_DIR.")
  ENDIF(ITK_FOUND)
ENDIF(USE_ITK)

INCLUDE (${CMAKE_ROOT}/Modules/FindITK.cmake) 
IF (USE_ITK_FILE) 
  INCLUDE(${USE_ITK_FILE}) 
ENDIF(USE_ITK_FILE) 

#SET(IJ_INCLUDE_DIRS
#)

#INCLUDE_DIRECTORIES(${IJ_INCLUDE_DIRS})

LINK_DIRECTORIES(${ITK_LIBRARY_PATH})

IF(BUILD_TESTING)
# non-templated class -- this should be stored in a library and linked in...
ADD_EXECUTABLE(itkN3MRIBiasFieldCorrectionImageFilterTest itkN3MRIBiasFieldCorrectionImageFilterTest.cxx)
TARGET_LINK_LIBRARIES(itkN3MRIBiasFieldCorrectionImageFilterTest ITKIO ITKStatistics ITKNumerics)

SET(CurrentExe "ImageCompare")
ADD_EXECUTABLE(${CurrentExe} ImageCompare.cxx) 
TARGET_LINK_LIBRARIES(${CurrentExe}  ITKIO ITKNumerics ITKStatistics)

SET(SOURCES "itkCommandLineParser" "itkCommandLineOption" )
SET(CurrentExe "N4" )
ADD_EXECUTABLE( ${CurrentExe} N4.cxx ${SOURCES} )
TARGET_LINK_LIBRARIES(${CurrentExe}  ITKIO ITKNumerics ITKStatistics)

ENDIF(BUILD_TESTING)

###
#  Perform testing
###

ADD_TEST( TEST_Brain2D itkN3MRIBiasFieldCorrectionImageFilterTest 2
        ${CMAKE_SOURCE_DIR}/Input/t81slice.nii.gz 
        ${CMAKE_BINARY_DIR}/t81corrected.nii.gz 
        2
        ${CMAKE_SOURCE_DIR}/Input/t81mask.nii.gz 
        50 4 
        ${CMAKE_BINARY_DIR}/t81biasfield.nii.gz )
#ADD_TEST( COMPARE_Brain2D ImageCompare
#        ${CMAKE_BINARY_DIR}/t81corrected.nii.gz 
#        ${CMAKE_SOURCE_DIR}/Output/t81corrected.nii.gz )
#ADD_TEST( COMPARE_Field2D ImageCompare
#        ${CMAKE_BINARY_DIR}/t81biasfield.nii.gz 
#        ${CMAKE_SOURCE_DIR}/Output/t81biasfield.nii.gz )

ADD_TEST( TEST_Lungs3D itkN3MRIBiasFieldCorrectionImageFilterTest 3
        ${CMAKE_SOURCE_DIR}/Input/he3volume.nii.gz 
        ${CMAKE_BINARY_DIR}/he3corrected.nii.gz 
        4  
        ${CMAKE_SOURCE_DIR}/Input/he3mask.nii.gz 
        50 4 
        ${CMAKE_BINARY_DIR}/he3biasfield.nii.gz )
#ADD_TEST( COMPARE_Lungs3D ImageCompare
#        ${CMAKE_BINARY_DIR}/he3corrected.nii.gz 
#        ${CMAKE_SOURCE_DIR}/Output/he3corrected.nii.gz )
#ADD_TEST( COMPARE_Field3D ImageCompare
#        ${CMAKE_BINARY_DIR}/he3biasfield.nii.gz 
#        ${CMAKE_SOURCE_DIR}/Output/he3biasfield.nii.gz )

ADD_TEST( TEST_Brain2D_N4 N4 2
        -i ${CMAKE_SOURCE_DIR}/Input/t81slice.nii.gz 
        -o [${CMAKE_BINARY_DIR}/t81corrected_n4.nii.gz,${CMAKE_BINARY_DIR}/t81biasfield_n4.nii.gz]
        -x ${CMAKE_SOURCE_DIR}/Input/t81mask.nii.gz
        -s 2
        -c [50x50x50,0.001]
        -b [100,3] 
        -t [0.15,0.01,200]
         )



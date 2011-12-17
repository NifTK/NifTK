# This is the root ITK CMakeLists file.
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)

# This project is designed to be built outside the Insight source tree.
PROJECT(BoundaryShiftIntegral)

# Find ITK.
FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

#ADD_EXECUTABLE(niftkBSI niftkBSI.cxx)
ADD_EXECUTABLE(niftkKN-BSI niftkKMeansWindowWithLinearRegressionNormalisationBSI.cxx)

#TARGET_LINK_LIBRARIES(niftkBSI ITKCommon ITKIO ITKBasicFilters ITKAlgorithms)
TARGET_LINK_LIBRARIES(niftkKN-BSI ITKCommon ITKIO ITKBasicFilters ITKAlgorithms)


INSTALL(TARGETS niftkKN-BSI RUNTIME DESTINATION bin)


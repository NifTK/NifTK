#-----------------------------------------------------------------------------
# GDCM
#
# Note: NifTK as such doesn't need GDCM. However, if we use MITK,
# then MITK needs a version of ITK that has been built with a specfic 
# version of GDCM. So we build GDCM, and then ITK in that same fashion.
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED GDCM_DIR AND NOT EXISTS ${GDCM_DIR})
  MESSAGE(FATAL_ERROR "GDCM_DIR variable is defined but corresponds to non-existing directory \"${GDCM_DIR}\".")
ENDIF()

SET(proj GDCM)
SET(proj_DEPENDENCIES )
SET(GDCM_DEPENDS ${proj})
SET(GDCM_VERSION 2.0.18)

IF(NOT DEFINED GDCM_DIR)
  ExternalProject_Add(${proj}
     URL http://cmic.cs.ucl.ac.uk/platform/dependencies/gdcm-${GDCM_VERSION}.tar.gz
     BINARY_DIR ${proj}-build
     INSTALL_COMMAND ""
     PATCH_COMMAND ${CMAKE_COMMAND} -DTEMPLATE_FILE:FILEPATH=${CMAKE_SOURCE_DIR}/CMake/CMakeExternals/EmptyFileForPatching.dummy -P ${CMAKE_SOURCE_DIR}/CMake/CMakeExternals/PatchGDCM-2.0.18.cmake
     CMAKE_GENERATOR ${GEN}
     CMAKE_ARGS
       ${EP_COMMON_ARGS}
       -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
       -DGDCM_BUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
       -DBUILD_TESTING:BOOL=${EP_BUILD_TESTING}
       -DBUILD_EXAMPLES:BOOL=${EP_BUILD_EXAMPLES}
     DEPENDS ${proj_DEPENDENCIES}
    )
  SET(GDCM_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build)
  MESSAGE("SuperBuild loading GDCM from ${GDCM_DIR}")

  SET(GDCM_IS_2_0_18 TRUE)
  
ELSE()

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")
  
  FIND_PACKAGE(GDCM)

  IF( GDCM_BUILD_VERSION EQUAL "18")
    SET(GDCM_IS_2_0_18 TRUE)
  ELSE()
    SET(GDCM_IS_2_0_18 FALSE)
  ENDIF()
 
ENDIF()

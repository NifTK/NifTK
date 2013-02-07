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


#-----------------------------------------------------------------------------
# ITK
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED ITK_DIR AND NOT EXISTS ${ITK_DIR})
  MESSAGE(FATAL_ERROR "ITK_DIR variable is defined but corresponds to non-existing directory \"${ITK_DIR}\".")
ENDIF()

SET(proj ITK)
SET(proj_DEPENDENCIES GDCM)
SET(ITK_DEPENDS ${proj})

IF(NOT DEFINED ITK_DIR)

  SET(additional_cmake_args )
  IF(MINGW)
    SET(additional_cmake_args
        -DCMAKE_USE_WIN32_THREADS:BOOL=ON
        -DCMAKE_USE_PTHREADS:BOOL=OFF)
  ENDIF()

  IF(GDCM_IS_2_0_18)
    IF(NIFTK_VERSION_ITK MATCHES "3.20.1")
      SET(ITK_PATCH_COMMAND ${CMAKE_COMMAND} -DTEMPLATE_FILE:FILEPATH=${CMAKE_SOURCE_DIR}/CMake/CMakeExternals/EmptyFileForPatching.dummy -P ${CMAKE_SOURCE_DIR}/CMake/CMakeExternals/PatchITK-3.20.cmake)
    ENDIF()
  ENDIF()

  niftkMacroGetChecksum(NIFTK_CHECKSUM_ITK ${NIFTK_LOCATION_ITK})

  ExternalProject_Add(${proj}
     URL ${NIFTK_LOCATION_ITK}
     URL_MD5 ${NIFTK_CHECKSUM_ITK}
     BINARY_DIR ${proj}-build
     INSTALL_COMMAND ""
     PATCH_COMMAND ${ITK_PATCH_COMMAND}
     CMAKE_GENERATOR ${GEN}
     CMAKE_ARGS
       ${EP_COMMON_ARGS}
       ${additional_cmake_args}
       -DBUILD_TESTING:BOOL=${EP_BUILD_TESTING}
       -DBUILD_EXAMPLES:BOOL=${EP_BUILD_EXAMPLES}
       -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
       -DITK_USE_SYSTEM_GDCM:BOOL=ON
       -DGDCM_DIR:PATH=${GDCM_DIR}
       -DITK_USE_REGION_VALIDATION_IN_ITERATORS:BOOL=OFF
       -DITK_USE_REVIEW:BOOL=ON
       -DITK_USE_REVIEW_STATISTICS:BOOL=OFF
       -DITK_USE_PATENTED:BOOL=ON
       -DITK_USE_OPTIMIZED_REGISTRATION_METHODS:BOOL=ON
       -DITK_USE_PORTABLE_ROUND:BOOL=ON
       -DITK_USE_CENTERED_PIXEL_COORDINATES_CONSISTENTLY:BOOL=ON
       -DITK_USE_TRANSFORM_IO_FACTORIES:BOOL=ON
       -DITK_LEGACY_REMOVE:BOOL=OFF
     DEPENDS ${proj_DEPENDENCIES}
  )
 
  SET(ITK_DIR ${CMAKE_CURRENT_BINARY_DIR}/${proj}-build)
  MESSAGE("SuperBuild loading ITK from ${ITK_DIR}")

ELSE(NOT DEFINED ITK_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

ENDIF(NOT DEFINED ITK_DIR)

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
# AprilTags - external project for tracking AR markers.
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED AprilTags_DIR AND NOT EXISTS ${AprilTags_DIR})
  message(FATAL_ERROR "AprilTags_DIR variable is defined but corresponds to non-existing directory \"${AprilTags_DIR}\".")
endif()

if(BUILD_NiftyIGI)

  set(version "3c6af59723")
  set(location "${NIFTK_EP_TARBALL_LOCATION}/NifTK-apriltags-${version}.tar.gz")
  set(depends OpenCV Eigen)

  niftkMacroDefineExternalProjectVariables(AprilTags ${version} ${location} "${depends}")

  if(NOT DEFINED AprilTags_DIR)

    if(UNIX)
      set(APRILTAGS_CXX_FLAGS "-fPIC")
      set(APRILTAGS_C_FLAGS "-fPIC")
    endif()

    ExternalProject_Add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${proj_LOCATION}
      URL_MD5 ${proj_CHECKSUM}
      UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${proj_VERSION}
      CMAKE_GENERATOR ${gen}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        -DBUILD_SHARED_LIBS:BOOL=OFF
        -DCMAKE_DEBUG_POSTFIX:STRING=
        -DEigen_INCLUDE_DIR:PATH=${Eigen_INCLUDE_DIR}
        "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS} ${APRILTAGS_CXX_FLAGS}"
        "-DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS} ${APRILTAGS_C_FLAGS}"
      CMAKE_CACHE_ARGS
        ${EP_COMMON_CACHE_ARGS}
      CMAKE_CACHE_DEFAULT_ARGS
        ${EP_COMMON_CACHE_DEFAULT_ARGS}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(AprilTags_DIR ${proj_INSTALL})

    mitkFunctionInstallExternalCMakeProject(${proj})

    message("SuperBuild loading AprilTags from ${AprilTags_DIR}")

  else(NOT DEFINED AprilTags_DIR)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED AprilTags_DIR)

endif()

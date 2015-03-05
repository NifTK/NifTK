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
# RTK
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED RTK_DIR AND NOT EXISTS ${RTK_DIR})
  message(FATAL_ERROR "RTK_DIR variable is defined but corresponds to non-existing directory \"${RTK_DIR}\".")
endif()

if(BUILD_RTK)

  set(version "196aec7d3b")
  set(location "${NIFTK_EP_TARBALL_LOCATION}/NifTK-RTK-${version}.tar.gz")

  niftkMacroDefineExternalProjectVariables(RTK ${version} ${location})
  set(proj_DEPENDENCIES GDCM ITK)

  if(NOT DEFINED RTK_DIR)

    set(additional_cmake_args )

    ExternalProject_Add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${proj_LOCATION}
      URL_MD5 ${proj_CHECKSUM}
      UPDATE_COMMAND  ${GIT_EXECUTABLE} checkout ${proj_VERSION}
      CMAKE_GENERATOR ${gen}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        ${additional_cmake_args}
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        -DITK_DIR:PATH=${ITK_DIR}
        -DGDCM_DIR:PATH=${GDCM_DIR}
        -DCMAKE_SHARED_LINKER_FLAGS:STRING=-L${GDCM_DIR}/bin
        -DCMAKE_EXE_LINKER_FLAGS:STRING=-L${GDCM_DIR}/bin
        -DBUILD_SHARED_LIBS:BOOL=OFF
      DEPENDS ${proj_DEPENDENCIES}
    )

    if(EP_ALWAYS_USE_INSTALL_DIR)
      set(RTK_DIR ${proj_INSTALL})
      set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})
    else()
      set(RTK_DIR ${proj_BUILD})
    endif()

    message("SuperBuild loading RTK from ${RTK_DIR}")

  else()

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif()
endif()

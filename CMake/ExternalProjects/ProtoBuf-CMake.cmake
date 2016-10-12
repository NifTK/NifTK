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
# ProtoBuf-CMake
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED ProtoBuf-CMake_DIR AND NOT EXISTS ${ProtoBuf-CMake_DIR})
  message(FATAL_ERROR "ProtoBuf-CMake_DIR variable is defined but corresponds to non-existing directory \"${ProtoBuf-CMake_ROOT}\".")
endif()

if(MITK_USE_ProtoBuf-CMake)

  set(version "9c19a2e")
  set(location "${NIFTK_EP_TARBALL_LOCATION}/protobuf-cmake-${version}.tar.gz")

  niftkMacroDefineExternalProjectVariables(ProtoBuf-CMake ${version} ${location})

  set(proj_DEPENDENCIES ProtoBuf)

  if(NOT DEFINED ProtoBuf-CMake_DIR)

    ExternalProject_Add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${ProtoBuf_BUILD_DIR}
      INSTALL_DIR ${ProtoBuf_DIR}
      URL ${proj_LOCATION}
      URL_MD5 ${proj_CHECKSUM}
      #CONFIGURE_COMMAND "${proj_SOURCE}/configure"
      #  "--prefix=${proj_INSTALL}"
      #UPDATE_COMMAND ""
      #BUILD_COMMAND ""
      #INSTALL_COMMAND ""
      CMAKE_GENERATOR ${gen}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        -DPROTOBUF_ROOT:PATH=${ProtoBuf_SOURCE_DIR}
      CMAKE_CACHE_ARGS
        ${EP_COMMON_CACHE_ARGS}
      CMAKE_CACHE_DEFAULT_ARGS
        ${EP_COMMON_CACHE_DEFAULT_ARGS}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(ProtoBuf-CMake_SOURCE_DIR ${proj_SOURCE})
    set(ProtoBuf-CMake_DIR ${proj_INSTALL})
    set(ProtoBuf-CMake_INCLUDE_DIR ${ProtoBuf-CMake_DIR}/include)
    set(ProtoBuf-CMake_LIBRARY_DIR ${ProtoBuf-CMake_DIR}/lib)

    mitkFunctionInstallExternalCMakeProject(${proj})

    message("SuperBuild loading ProtoBuf-CMake from ${ProtoBuf-CMake_DIR}.")

  else(NOT DEFINED ProtoBuf-CMake_DIR)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED ProtoBuf-CMake_DIR)

endif()

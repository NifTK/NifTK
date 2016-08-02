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
# ProtoBuf
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED ProtoBuf_DIR AND NOT EXISTS ${ProtoBuf_DIR})
  message(FATAL_ERROR "ProtoBuf_DIR variable is defined but corresponds to non-existing directory \"${ProtoBuf_ROOT}\".")
endif()

set(version "2.6.1")
set(location "${NIFTK_EP_TARBALL_LOCATION}/protobuf-${version}.tar.gz")

niftkMacroDefineExternalProjectVariables(ProtoBuf ${version} ${location})

if(NOT DEFINED ProtoBuf_DIR)

  ExternalProject_Add(${proj}
    LIST_SEPARATOR ^^
    PREFIX ${proj_CONFIG}
    SOURCE_DIR ${proj_SOURCE}
    BINARY_DIR ${proj_BUILD}
    INSTALL_DIR ${proj_INSTALL}
    URL ${proj_LOCATION}
    URL_MD5 ${proj_CHECKSUM}
    CONFIGURE_COMMAND "${proj_SOURCE}/configure"
      "--prefix=${proj_INSTALL}"
    #UPDATE_COMMAND ""
    #BUILD_COMMAND ""
    #INSTALL_COMMAND ""
    CMAKE_GENERATOR ${gen}
    CMAKE_ARGS
      ${EP_COMMON_ARGS}
      -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
    CMAKE_CACHE_ARGS
      ${EP_COMMON_CACHE_ARGS}
    CMAKE_CACHE_DEFAULT_ARGS
      ${EP_COMMON_CACHE_DEFAULT_ARGS}
    DEPENDS ${proj_DEPENDENCIES}
  )

  set(ProtoBuf_SOURCE_DIR ${proj_SOURCE})
  set(ProtoBuf_DIR ${proj_INSTALL})
  set(ProtoBuf_INCLUDE_DIR ${ProtoBuf_DIR}/include)
  set(ProtoBuf_LIBRARY_DIR ${ProtoBuf_DIR}/lib)

  mitkFunctionInstallExternalCMakeProject(${proj})

  message("SuperBuild loading ProtoBuf from ${ProtoBuf_DIR}.")

else(NOT DEFINED ProtoBuf_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif(NOT DEFINED ProtoBuf_DIR)

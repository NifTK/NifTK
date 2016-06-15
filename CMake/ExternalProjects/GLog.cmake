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
# GLog
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED GLog_DIR AND NOT EXISTS ${GLog_DIR})
  message(FATAL_ERROR "GLog_DIR variable is defined but corresponds to non-existing directory \"${GLog_ROOT}\".")
endif()

set(version "0.3.3")
set(location "${NIFTK_EP_TARBALL_LOCATION}/glog-${version}.tar.gz")

niftkMacroDefineExternalProjectVariables(GLog ${version} ${location})

if(NOT DEFINED GLog_DIR)

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

  set(GLog_SOURCE_DIR ${proj_SOURCE})
  set(GLog_DIR ${proj_INSTALL})
  set(GLog_INCLUDE_DIR ${GLog_DIR}/include)
  set(GLog_LIBRARY_DIR ${GLog_DIR}/lib)

  # Needed by Caffe
  find_library(GLog_LIBRARY_RELEASE NAMES glog
               PATHS ${GLog_DIR}
               PATH_SUFFIXES lib lib/Release
               NO_DEFAULT_PATH)
  find_library(GLog_LIBRARY_DEBUG NAMES glogd
               PATHS ${GLog_DIR}
               PATH_SUFFIXES lib lib/Debug
               NO_DEFAULT_PATH)

  set(GLog_LIBRARY )
  if(GLog_LIBRARY_RELEASE)
    list(APPEND GLog_LIBRARY ${GLog_LIBRARY_RELEASE})
  endif()
  if(GLog_LIBRARY_DEBUG)
    list(APPEND GLog_LIBRARY ${GLog_LIBRARY_DEBUG})
  endif()

  mitkFunctionInstallExternalCMakeProject(${proj})

  message("SuperBuild loading GLog from ${GLog_DIR}.")

else(NOT DEFINED GLog_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif(NOT DEFINED GLog_DIR)

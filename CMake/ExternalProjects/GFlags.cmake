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
# GFlags
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED GFlags_DIR AND NOT EXISTS ${GFlags_DIR})
  message(FATAL_ERROR "GFlags_DIR variable is defined but corresponds to non-existing directory \"${GFlags_ROOT}\".")
endif()

set(GFlags_VERSION "2.1.2")
set(location "${NIFTK_EP_TARBALL_LOCATION}/gflags-${GFlags_VERSION}.tar.gz")

niftkMacroDefineExternalProjectVariables(GFlags ${GFlags_VERSION} ${location})

if(NOT DEFINED GFlags_DIR)

  ExternalProject_Add(${proj}
    LIST_SEPARATOR ^^
    PREFIX ${proj_CONFIG}
    SOURCE_DIR ${proj_SOURCE}
    BINARY_DIR ${proj_BUILD}
    INSTALL_DIR ${proj_INSTALL}
    URL ${proj_LOCATION}
    URL_MD5 ${proj_CHECKSUM}
    #CONFIGURE_COMMAND ""
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

  #set(GFlags_SOURCE_DIR ${proj_SOURCE})
  set(GFlags_DIR ${proj_INSTALL})
  set(GFlags_INCLUDE_DIR ${GFlags_DIR}/include)
  set(GFlags_LIBRARY_DIR ${GFlags_DIR}/lib)
  
  mitkFunctionInstallExternalCMakeProject(${proj})

  message("SuperBuild loading GFlags from ${GFlags_DIR}.")

else(NOT DEFINED GFlags_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif(NOT DEFINED GFlags_DIR)

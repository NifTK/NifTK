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
# OpenBLAS
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED OpenBLAS_DIR AND NOT EXISTS ${OpenBLAS_DIR})
  message(FATAL_ERROR "OpenBLAS_DIR variable is defined but corresponds to non-existing directory \"${OpenBLAS_ROOT}\".")
endif()

if(NOT APPLE)

  #set(version "0.2.18")
  set(version "7daf34e")
  set(location "${NIFTK_EP_TARBALL_LOCATION}/OpenBLAS-${version}.tar.gz")

  niftkMacroDefineExternalProjectVariables(OpenBLAS ${version} ${location})

  if(NOT DEFINED OpenBLAS_DIR)

    if(WIN32)
      set(OpenBLAS_USE_THREAD OFF)
    else(WIN32)    
      set(OpenBLAS_USE_THREAD ON)
    endif(WIN32)

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
      INSTALL_COMMAND ""
      CMAKE_GENERATOR ${gen}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
        -DUSE_THREAD:BOOL=${OpenBLAS_USE_THREAD}
        -DCMAKE_DEBUG_POSTFIX:STRING=
     CMAKE_CACHE_ARGS
        ${EP_COMMON_CACHE_ARGS}
      CMAKE_CACHE_DEFAULT_ARGS
        ${EP_COMMON_CACHE_DEFAULT_ARGS}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(OpenBLAS_SOURCE_DIR ${proj_SOURCE})
    set(OpenBLAS_DIR        ${proj_BUILD})
    set(OpenBLAS_INCLUDE_DIR ${OpenBLAS_SOURCE_DIR})
    set(OpenBLAS_LIBRARY_DIR ${OpenBLAS_DIR}/lib)

    install(DIRECTORY ${proj_BUILD}/include
            DESTINATION ${proj_INSTALL})

    install(DIRECTORY ${proj_BUILD}/lib
            DESTINATION ${proj_INSTALL})

    mitkFunctionInstallExternalCMakeProject(${proj})

    message("SuperBuild loading OpenBLAS from ${OpenBLAS_DIR}.")

  else(NOT DEFINED OpenBLAS_DIR)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED OpenBLAS_DIR)
endif()



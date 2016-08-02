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
# HDF5
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED HDF5_DIR AND NOT EXISTS ${HDF5_DIR})
  message(FATAL_ERROR "HDF5_DIR variable is defined but corresponds to non-existing directory \"${HDF5_ROOT}\".")
endif()

set(version "1.10.0-patch1")
set(location "${NIFTK_EP_TARBALL_LOCATION}/hdf5-${version}.tar.gz")

niftkMacroDefineExternalProjectVariables(HDF5 ${version} ${location})

if(NOT DEFINED HDF5_DIR)

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
      --enable-cxx
      -DCMAKE_BUILD_TYPE=Release
    CMAKE_CACHE_ARGS
      ${EP_COMMON_CACHE_ARGS}
    CMAKE_CACHE_DEFAULT_ARGS
      ${EP_COMMON_CACHE_DEFAULT_ARGS}
    DEPENDS ${proj_DEPENDENCIES}
  )

  set(HDF5_SOURCE_DIR ${proj_SOURCE})
  set(HDF5_DIR ${proj_INSTALL})
  set(HDF5_INCLUDE_DIR ${HDF5_DIR}/include)
  set(HDF5_LIBRARY_DIR ${HDF5_DIR}/lib)

  # If using: cmake-3.2.3/share/cmake-3.2/Modules/FindHDF5.cmake
  set(ENV{HDF5_ROOT_DIR_HINT} ${HDF5_DIR}/share/cmake)  
  set(ENV{HDF5_ROOT} ${HDF5_DIR})  

  mitkFunctionInstallExternalCMakeProject(${proj})

  message("SuperBuild loading HDF5 from ${HDF5_DIR}.")

else(NOT DEFINED HDF5_DIR)

  mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

endif(NOT DEFINED HDF5_DIR)

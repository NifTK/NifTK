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
  message(FATAL_ERROR "HDF5_DIR variable is defined but corresponds to non-existing directory \"${HDF5_DIR}\".")
endif()

if(MITK_USE_HDF5)

  set(HDF5_VERSION "1.10.0-patch1")
  set(location "${NIFTK_EP_TARBALL_LOCATION}/hdf5-${HDF5_VERSION}.tar.gz")

  niftkMacroDefineExternalProjectVariables(HDF5 ${HDF5_VERSION} ${location})

  if(NOT DEFINED HDF5_DIR)

    ExternalProject_Add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${proj_LOCATION}
      URL_MD5 ${proj_CHECKSUM}
      CMAKE_GENERATOR ${gen}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        -DBUILD_EXAMPLES:BOOL=OFF
        -DBUILD_TOOLS:BOOL=OFF
        -DHDF5_BUILD_EXAMPLES:BOOL=OFF
        -DHDF5_EXTERNAL_LIB_PREFIX:String=niftk
        -DHDF5_EXTERNALLY_CONFIGURED:BOOL=ON
        --enable-cxx
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
    set(HDF5_HL_INCLUDE_DIR ${HDF5_INCLUDE_DIR})
    set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})

    mitkFunctionInstallExternalCMakeProject(${proj})

    message("SuperBuild loading HDF5 from ${HDF5_DIR}.")

  else(NOT DEFINED HDF5_DIR)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED HDF5_DIR)

endif()

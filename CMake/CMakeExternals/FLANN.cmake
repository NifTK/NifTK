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
# FLANN - external project needed by PCL.
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED FLANN_DIR AND NOT EXISTS ${FLANN_DIR})
  message(FATAL_ERROR "FLANN_DIR variable is defined but corresponds to non-existing directory \"${FLANN_DIR}\".")
endif()

if(BUILD_IGI AND BUILD_PCL)

  set(version "1.8.4.1")
  set(location "${NIFTK_EP_TARBALL_LOCATION}/flann-${version}-src.tar.gz")

  niftkMacroDefineExternalProjectVariables(FLANN ${version} ${location})

  if(NOT DEFINED FLANN_DIR)

    ExternalProject_Add(${proj}
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${proj_LOCATION}
      URL_MD5 ${proj_CHECKSUM}
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        ${EP_COMMON_ARGS}
        -DBUILD_MATLAB_BINDINGS:BOOL=OFF
        -DBUILD_PYTHON_BINDINGS:BOOL=OFF
        -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
      DEPENDS ${proj_DEPENDENCIES}
    )

    set(FLANN_DIR ${proj_INSTALL})
    set(FLANN_ROOT ${FLANN_DIR})

    message("SuperBuild loading FLANN from ${FLANN_DIR}")

  else(NOT DEFINED FLANN_DIR)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED FLANN_DIR)

endif()

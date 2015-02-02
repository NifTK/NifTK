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

  niftkMacroDefineExternalProjectVariables(FLANN ${NIFTK_VERSION_FLANN})

  if(NOT DEFINED FLANN_DIR)

    niftkMacroGetChecksum(NIFTK_CHECKSUM_FLANN ${NIFTK_LOCATION_FLANN})

    ExternalProject_Add(${proj}
      SOURCE_DIR ${proj_SOURCE}
      PREFIX ${proj_CONFIG}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      URL ${NIFTK_LOCATION_FLANN}
      URL_MD5 ${NIFTK_CHECKSUM_FLANN}
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

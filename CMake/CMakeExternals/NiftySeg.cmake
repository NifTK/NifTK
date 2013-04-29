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
# NIFTYSEG
#-----------------------------------------------------------------------------

# Sanity checks
IF(DEFINED NIFTYSEG_ROOT AND NOT EXISTS ${NIFTYSEG_ROOT})
  MESSAGE(FATAL_ERROR "NIFTYSEG_ROOT variable is defined but corresponds to non-existing disegtory \"${NIFTYSEG_ROOT}\".")
ENDIF()

IF(BUILD_NIFTYSEG)

  SET(proj NIFTYSEG)
  SET(proj_DEPENDENCIES )
  SET(proj_INSTALL ${EP_BASE}/Install/${proj} )
  SET(NIFTYSEG_DEPENDS ${proj})

  IF(NOT DEFINED NIFTYSEG_ROOT)

    niftkMacroGetChecksum(NIFTK_CHECKSUM_NIFTYSEG ${NIFTK_LOCATION_NIFTYSEG})

    ExternalProject_Add(${proj}
      URL ${NIFTK_LOCATION_NIFTYSEG}
      URL_MD5 ${NIFTK_CHECKSUM_NIFTYSEG}
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
        -DUSE_CUDA:BOOL=${NIFTK_USE_CUDA}
        -DUSE_OPENMP:BOOL=OFF
        -DINSTALL_PRIORS:BOOL=ON
        -DINSTALL_PRIORS_DIRECTORY:PATH=${EP_BASE}/Install/${proj}/priors
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
      DEPENDS ${proj_DEPENDENCIES}
      )

    SET(NIFTYSEG_ROOT ${proj_INSTALL})
    SET(NIFTYSEG_INCLUDE_DIR "${NIFTYSEG_ROOT}/include")
    SET(NIFTYSEG_LIBRARY_DIR "${NIFTYSEG_ROOT}/lib")

    MESSAGE("SuperBuild loading NIFTYSEG from ${NIFTYSEG_ROOT}")

  ELSE(NOT DEFINED NIFTYSEG_ROOT)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  ENDIF(NOT DEFINED NIFTYSEG_ROOT)

ENDIF(BUILD_NIFTYSEG)

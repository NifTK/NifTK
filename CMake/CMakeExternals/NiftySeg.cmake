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
# NiftySeg
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED NIFTYSEG_ROOT AND NOT EXISTS ${NIFTYSEG_ROOT})
  message(FATAL_ERROR "NIFTYSEG_ROOT variable is defined but corresponds to non-existing disegtory \"${NIFTYSEG_ROOT}\".")
endif()

if(BUILD_NIFTYSEG)

  set(proj NiftySeg)
  set(proj_DEPENDENCIES )
  set(proj_INSTALL ${CMAKE_BINARY_DIR}/${proj}-install )
  set(NIFTYSEG_DEPENDS ${proj})

  if(NOT DEFINED NIFTYSEG_ROOT)

    niftkMacroGetChecksum(NIFTK_CHECKSUM_NIFTYSEG ${NIFTK_LOCATION_NIFTYSEG})

    ExternalProject_Add(${proj}
      SOURCE_DIR ${proj}-src
      BINARY_DIR ${proj}-build
      PREFIX ${proj}-cmake
      INSTALL_DIR ${proj}-install
      URL ${NIFTK_LOCATION_NIFTYSEG}
      URL_MD5 ${NIFTK_CHECKSUM_NIFTYSEG}
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
        -DUSE_CUDA:BOOL=${NIFTK_USE_CUDA}
        -DUSE_OPENMP:BOOL=OFF
        -DINSTALL_PRIORS:BOOL=ON
        -DINSTALL_PRIORS_DIRECTORY:PATH=${proj_INSTALL}/priors
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
      DEPENDS ${proj_DEPENDENCIES}
      )

    set(NIFTYSEG_ROOT ${proj_INSTALL})
    set(NIFTYSEG_INCLUDE_DIR "${NIFTYSEG_ROOT}/include")
    set(NIFTYSEG_LIBRARY_DIR "${NIFTYSEG_ROOT}/lib")

    message("SuperBuild loading NiftySeg from ${NIFTYSEG_ROOT}")

  else(NOT DEFINED NIFTYSEG_ROOT)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED NIFTYSEG_ROOT)

endif(BUILD_NIFTYSEG)

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
# NiftyPipe
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED NIFTYPIPE_ROOT AND NOT EXISTS ${NIFTYPIPE_ROOT})
  message(FATAL_ERROR "NIFTYPIPE_ROOT variable is defined but corresponds to non-existing directory \"${NIFTYPIPE_ROOT}\".")
endif()

if(BUILD_NIFTYPIPE)

  set(proj NiftyPipe)
  set(proj_INSTALL ${CMAKE_BINARY_DIR}/${proj}-install )
  set(NIFTYPIPE_DEPENDS ${proj})

  if(NOT DEFINED NIFTYPIPE_ROOT)

    ExternalProject_Add(${proj}
      SOURCE_DIR ${proj}-src
      BINARY_DIR ${proj}-build
      PREFIX ${proj}-cmake
      INSTALL_DIR ${proj}-install
      GIT_REPOSITORY ${NIFTK_LOCATION_NiftyPipe_GIT}
      GIT_TAG ${NIFTK_VERSION_NiftyPipe}
      UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${NIFTK_VERSION_NIFTYPIPE}
      CMAKE_GENERATOR ${GEN}
      CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
      )

    set(NIFTYPIPE_ROOT ${proj_INSTALL})

    message("SuperBuild loading NiftyPipe from ${NIFTYPIPE_ROOT}")

  else(NOT DEFINED NIFTYPIPE_ROOT)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED NIFTYPIPE_ROOT)

endif(BUILD_NIFTYPIPE)

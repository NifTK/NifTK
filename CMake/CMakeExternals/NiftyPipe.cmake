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

  set(version "5766ac61c9")
  set(location "git@cmiclab.cs.ucl.ac.uk:CMIC/NiftyPipe")

  niftkMacroDefineExternalProjectVariables(NiftyPipe ${version} ${location})

  if(NOT DEFINED NIFTYPIPE_ROOT)

    ExternalProject_Add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      GIT_REPOSITORY ${proj_LOCATION}
      GIT_TAG ${proj_VERSION}
      UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${proj_VERSION}
      CMAKE_GENERATOR ${gen}
      CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
      )

    set(NIFTYPIPE_ROOT ${proj_INSTALL})

    message("SuperBuild loading NiftyPipe from ${NIFTYPIPE_ROOT}")

  else(NOT DEFINED NIFTYPIPE_ROOT)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED NIFTYPIPE_ROOT)

endif(BUILD_NIFTYPIPE)

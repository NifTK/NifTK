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
if(DEFINED NiftyPipe_ROOT AND NOT EXISTS ${NiftyPipe_ROOT})
  message(FATAL_ERROR "NiftyPipe_ROOT variable is defined but corresponds to non-existing directory \"${NiftyPipe_ROOT}\".")
endif()

if(BUILD_NiftyPipe)
  
  set(version "d00c0d7302")
  set(location "git@cmiclab.cs.ucl.ac.uk:CMIC/NiftyPipe")

  niftkMacroDefineExternalProjectVariables(NiftyPipe ${version} ${location})

  if(NOT DEFINED NiftyPipe_ROOT)

    ExternalProject_Add(${proj}
      LIST_SEPARATOR ^^
      PREFIX ${proj_CONFIG}
      SOURCE_DIR ${proj_SOURCE}
      BINARY_DIR ${proj_BUILD}
      INSTALL_DIR ${proj_INSTALL}
      GIT_REPOSITORY ${proj_LOCATION}
      GIT_TAG ${proj_VERSION}
      CMAKE_GENERATOR ${gen}
      CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
      )

    set(NiftyPipe_ROOT ${proj_INSTALL})

    message("SuperBuild loading NiftyPipe from ${NiftyPipe_ROOT}")

  else(NOT DEFINED NiftyPipe_ROOT)

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif(NOT DEFINED NiftyPipe_ROOT)

endif(BUILD_NiftyPipe)

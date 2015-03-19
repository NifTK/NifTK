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
# VL
#-----------------------------------------------------------------------------

# Sanity checks
if(DEFINED VL_ROOT AND NOT EXISTS ${VL_ROOT})
  message(FATAL_ERROR "VL_ROOT variable is defined but corresponds to non-existing directory \"${VL_ROOT}\".")
endif()

if(BUILD_VL)

  set(version "624a035fb7")
  set(location "https://cmiclab.cs.ucl.ac.uk/CMIC/VisualizationLibrary.git")
  
  niftkMacroDefineExternalProjectVariables(VL ${version} ${location})

  if(NOT DEFINED VL_DIR)

    set(additional_cmake_args )

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
        ${EP_COMMON_ARGS}
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
        -DBUILD_TESTING:BOOL=${EP_BUILD_TESTING}
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
        -DVL_GUI_QT4_SUPPORT:BOOL=${QT_FOUND}
        ${additional_cmake_args}
      DEPENDS ${proj_DEPENDENCIES}
    )

	set(VL_ROOT ${proj_INSTALL})
    set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})

    message("SuperBuild loading VL from ${VL_ROOT}")

  else()

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif()
endif()

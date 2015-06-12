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

  set(version "19b0375cbe")
  set(location "https://cmiclab.cs.ucl.ac.uk/CMIC/VisualizationLibrary.git")
  
  niftkMacroDefineExternalProjectVariables(VL ${version} ${location})

  if(NOT DEFINED VL_DIR)

    set(VL_ROOT ${proj_INSTALL})

    if(APPLE)
      set(APPLE_CMAKE_SCRIPT ${proj_CONFIG}/ChangeVLLibsInstallNameForMac.cmake)
      configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CMake/CMakeExternals/ChangeVLLibsInstallNameForMac.cmake.in ${APPLE_CMAKE_SCRIPT} @ONLY)
      set(APPLE_TEST_COMMAND ${CMAKE_COMMAND} -P ${APPLE_CMAKE_SCRIPT})
    endif()

    set(additional_cmake_args
        -DCMAKE_PREFIX_PATH:PATH=${NifTK_PREFIX_PATH}
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS:BOOL=${EP_BUILD_SHARED_LIBS}
        -DBUILD_TESTING:BOOL=${EP_BUILD_TESTING}
        -DCMAKE_INSTALL_PREFIX:PATH=${proj_INSTALL}
        -DVL_GUI_QT4_SUPPORT:BOOL=${QT_FOUND}
    )

    if (APPLE)
      ExternalProject_Add(${proj}
        LIST_SEPARATOR ^^
        PREFIX ${proj_CONFIG}
        SOURCE_DIR ${proj_SOURCE}
        BINARY_DIR ${proj_BUILD}
        INSTALL_DIR ${proj_INSTALL}
        GIT_REPOSITORY ${proj_LOCATION}
        GIT_TAG ${proj_VERSION}
        UPDATE_COMMAND ${GIT_EXECUTABLE} checkout ${proj_VERSION}
        TEST_AFTER_INSTALL 1
        TEST_COMMAND ${APPLE_TEST_COMMAND}
        CMAKE_GENERATOR ${gen}
        CMAKE_ARGS
          ${EP_COMMON_ARGS}
          ${additional_cmake_args}
        DEPENDS ${proj_DEPENDENCIES}
      )
    else()
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
          ${additional_cmake_args}
        DEPENDS ${proj_DEPENDENCIES}
      )
    endif()

    set(NifTK_PREFIX_PATH ${proj_INSTALL}^^${NifTK_PREFIX_PATH})

    message("SuperBuild loading VL from ${VL_ROOT}")

  else()

    mitkMacroEmptyExternalProject(${proj} "${proj_DEPENDENCIES}")

  endif()
endif()
